"""Train a small Qwen3 LoRA adapter on approved conversation records."""

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from clean_model.data import (
    build_supervised_examples,
    dataset_fingerprint,
    load_jsonl,
    split_by_group,
)


MODEL_ID = 'Qwen/Qwen3-0.6B'
MODEL_REVISION = 'c1899de289a04d12100db370d81485cdf75e47ca'
SEED = 7
MAX_SEQUENCE_LENGTH = 512
MAX_NEW_TOKENS = 96


def encode_assistant_turn(
    context: list[dict[str, str]],
    target: str,
    tokenizer: Any,
    max_sequence_length: int = MAX_SEQUENCE_LENGTH,
) -> dict[str, list[int]]:
    """Build a chat-template sequence with loss only on the assistant turn."""
    if not context or context[-1].get('role') != 'user':
        raise ValueError('assistant-turn context must end with a user message')
    if not target.strip():
        raise ValueError('assistant target must not be empty')

    prompt_ids = tokenizer.apply_chat_template(
        context,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    complete_ids = tokenizer.apply_chat_template(
        [*context, {'role': 'assistant', 'content': target}],
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    if complete_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError(
            'chat template prompt tokens do not match the full conversation')
    if len(complete_ids) > max_sequence_length:
        raise ValueError(
            'encoded conversation exceeds max_sequence_length=%d; shorten '
            'the reviewed example or increase the limit' % max_sequence_length)
    if len(complete_ids) <= len(prompt_ids):
        raise ValueError('assistant target produced no trainable tokens')

    labels = [-100] * len(prompt_ids) + complete_ids[len(prompt_ids):]
    return {'input_ids': complete_ids, 'labels': labels}


def prepare_splits(
    dataset_path: str | Path,
    seed: int = SEED,
    max_sequence_length: int = MAX_SEQUENCE_LENGTH,
    tokenizer: Any | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Validate approved data and encode group-safe train/validation/test turns."""
    records = load_jsonl(dataset_path, require_approved=True)
    if any(record['kind'] != 'dialogue' for record in records):
        raise ValueError('Qwen LoRA pilot accepts dialogue records only')
    split_records = split_by_group(records, seed=seed)

    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            use_fast=True,
        )
    if tokenizer.chat_template is None:
        raise ValueError('Qwen3 tokenizer does not provide a chat template')

    encoded_splits: dict[str, list[dict[str, Any]]] = {}
    for split_name, split in split_records.items():
        examples = []
        for item in build_supervised_examples(split, require_approved=True):
            encoded = encode_assistant_turn(
                item['context'],
                item['target'],
                tokenizer,
                max_sequence_length=max_sequence_length,
            )
            examples.append({
                **encoded,
                'record_id': item['record_id'],
                'group_id': item['group_id'],
            })
        if split_name in ('train', 'validation') and not examples:
            raise ValueError('%s split contains no assistant turns' % split_name)
        encoded_splits[split_name] = examples

    summary = {
        'dataset_sha256': _sha256(Path(dataset_path)),
        'dataset_fingerprint': dataset_fingerprint(records),
        'record_counts': {
            name: len(rows) for name, rows in split_records.items()
        },
        'assistant_turn_counts': {
            name: len(rows) for name, rows in encoded_splits.items()
        },
        'group_counts': {
            name: len({record['group_id'] for record in rows})
            for name, rows in split_records.items()
        },
    }
    return encoded_splits, summary


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _model_dtype(torch: Any) -> Any:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8 and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def train_adapter(
    dataset_path: str | Path,
    output_dir: str | Path,
    epochs: int = 8,
    batch_size: int = 2,
    learning_rate: float = 2e-4,
    patience: int = 3,
    seed: int = SEED,
) -> dict[str, Any]:
    """Train LoRA on train turns, select by validation loss, and save adapter."""
    if epochs < 1 or batch_size < 1 or patience < 1:
        raise ValueError('epochs, batch_size, and patience must be positive')
    if learning_rate <= 0:
        raise ValueError('learning_rate must be positive')
    output_path = Path(output_dir)

    import torch
    from peft import LoraConfig, get_peft_model
    from torch.nn.utils.rnn import pad_sequence
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required; use a Kaggle GPU accelerator')
    output_path.mkdir(parents=True, exist_ok=False)
    _set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    splits, data_summary = prepare_splits(
        dataset_path,
        seed=seed,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        tokenizer=tokenizer,
    )
    dtype = _model_dtype(torch)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    base_model.config.use_cache = False
    model = get_peft_model(
        base_model,
        LoraConfig(
            task_type='CAUSAL_LM',
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=[
                'q_proj', 'k_proj', 'v_proj', 'o_proj',
                'gate_proj', 'up_proj', 'down_proj',
            ],
        ),
    )
    model.print_trainable_parameters()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters()
         if parameter.requires_grad),
        lr=learning_rate,
    )
    scaler = torch.amp.GradScaler(
        'cuda',
        enabled=dtype == torch.float16,
    )
    pad_id = tokenizer.pad_token_id

    def collate(rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        inputs = [
            torch.tensor(row['input_ids'], dtype=torch.long)
            for row in rows
        ]
        labels = [
            torch.tensor(row['labels'], dtype=torch.long)
            for row in rows
        ]
        return {
            'input_ids': pad_sequence(
                inputs, batch_first=True, padding_value=pad_id),
            'attention_mask': pad_sequence(
                [torch.ones_like(item) for item in inputs],
                batch_first=True,
                padding_value=0,
            ),
            'labels': pad_sequence(
                labels, batch_first=True, padding_value=-100),
        }

    def batches(
        rows: list[dict[str, Any]],
        shuffle: bool,
    ):
        order = list(range(len(rows)))
        if shuffle:
            random.shuffle(order)
        for start in range(0, len(order), batch_size):
            selected = [rows[index] for index in order[start:start + batch_size]]
            yield collate(selected)

    def evaluate_loss(rows: list[dict[str, Any]]) -> float:
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        with torch.inference_mode():
            for batch in batches(rows, shuffle=False):
                batch = {key: value.to('cuda') for key, value in batch.items()}
                result = model(**batch)
                token_count = int(batch['labels'].ne(-100).sum().item())
                total_loss += float(result.loss.item()) * token_count
                total_tokens += token_count
        if total_tokens == 0:
            raise ValueError('validation split contains no target tokens')
        return total_loss / total_tokens

    train_rows = splits['train']
    validation_rows = splits['validation']
    best_loss = float('inf')
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_total = 0.0
        train_token_total = 0
        for batch in batches(train_rows, shuffle=True):
            batch = {key: value.to('cuda') for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            result = model(**batch)
            scaler.scale(result.loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                (parameter for parameter in model.parameters()
                 if parameter.requires_grad),
                max_norm=1.0,
            )
            scaler.step(optimizer)
            scaler.update()
            token_count = int(batch['labels'].ne(-100).sum().item())
            train_loss_total += float(result.loss.detach().item()) * token_count
            train_token_total += token_count

        train_loss = train_loss_total / train_token_total
        validation_loss = evaluate_loss(validation_rows)
        epoch_result = {
            'epoch': epoch,
            'train_response_loss': train_loss,
            'validation_response_loss': validation_loss,
        }
        history.append(epoch_result)
        print(
            'epoch %d/%d train_response_loss=%.4f '
            'validation_response_loss=%.4f'
            % (epoch, epochs, train_loss, validation_loss))
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            model.save_pretrained(output_path)
            tokenizer.save_pretrained(output_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    metadata = {
        'experiment': 'qwen3_0_6b_dialogue_lora',
        'model_id': MODEL_ID,
        'model_revision': MODEL_REVISION,
        'seed': seed,
        'device': torch.cuda.get_device_name(0),
        'dtype': str(dtype),
        'lora': {
            'rank': 8,
            'alpha': 16,
            'dropout': 0.05,
            'target_modules': [
                'q_proj', 'k_proj', 'v_proj', 'o_proj',
                'gate_proj', 'up_proj', 'down_proj',
            ],
        },
        'training': {
            'epochs_requested': epochs,
            'epochs_completed': len(history),
            'best_epoch': best_epoch,
            'best_validation_response_loss': best_loss,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'patience': patience,
            'max_sequence_length': MAX_SEQUENCE_LENGTH,
            'history': history,
        },
        'data': data_summary,
        'held_out_test_evaluated': False,
        'limitation': (
            '20 owner-approved, project-authored examples are a tiny pilot; '
            'validation loss does not establish conversational quality.'),
    }
    (output_path / 'training_metadata.json').write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    return metadata


def evaluate_test_once(
    dataset_path: str | Path,
    adapter_dir: str | Path,
    report_path: str | Path,
    seed: int = SEED,
) -> dict[str, Any]:
    """Compare base and adapter on held-out test prompts exactly once."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required for held-out test evaluation')
    report_file = Path(report_path)
    if report_file.exists():
        raise FileExistsError(
            'refusing to overwrite one-shot test report: %s' % report_file)
    records = load_jsonl(dataset_path, require_approved=True)
    split_records = split_by_group(records, seed=seed)
    test_examples = list(build_supervised_examples(
        split_records['test'], require_approved=True))
    if not test_examples:
        raise ValueError('held-out test split contains no assistant turns')

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = _model_dtype(torch)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to('cuda')
    base_model.config.use_cache = True

    def generate(model: Any, item: dict[str, Any]) -> str:
        tokens = tokenizer.apply_chat_template(
            item['context'],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors='pt',
        ).to('cuda')
        attention_mask = torch.ones_like(tokens)
        with torch.inference_mode():
            output = model.generate(
                input_ids=tokens,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        return tokenizer.decode(
            output[0, tokens.shape[1]:],
            skip_special_tokens=True,
        ).strip()

    base_model.eval()
    base_responses = [
        generate(base_model, item)
        for item in test_examples
    ]
    tuned_model = PeftModel.from_pretrained(base_model, adapter_dir)
    tuned_model.eval()
    adapter_responses = [
        generate(tuned_model, item)
        for item in test_examples
    ]

    comparisons = []
    for index, item in enumerate(test_examples):
        comparisons.append({
            'record_id': item['record_id'],
            'group_id': item['group_id'],
            'context': item['context'],
            'reference': item['target'],
            'generations': [
                {'model': 'base', 'response': base_responses[index]},
                {'model': 'adapter', 'response': adapter_responses[index]},
            ],
        })

    report = {
        'experiment': 'qwen3_0_6b_dialogue_lora',
        'model_id': MODEL_ID,
        'model_revision': MODEL_REVISION,
        'seed': seed,
        'data_fingerprint': dataset_fingerprint(records),
        'test_record_count': len(split_records['test']),
        'test_turn_count': len(comparisons),
        'generation': {
            'strategy': 'greedy',
            'max_new_tokens': MAX_NEW_TOKENS,
            'enable_thinking': False,
        },
        'comparisons': comparisons,
        'limitation': (
            'Only two held-out conversations are available. This is a '
            'one-shot qualitative comparison, not a reliable quality score.'),
    }
    report_file.parent.mkdir(parents=True, exist_ok=True)
    with report_file.open('x', encoding='utf-8') as destination:
        json.dump(report, destination, ensure_ascii=False, indent=2)
        destination.write('\n')
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Train Qwen3-0.6B LoRA from approved Turkish dialogues')
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--learning-rate', type=float, default=2e-4)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()
    metadata = train_adapter(
        args.data,
        args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        seed=args.seed,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
