import os
import json
import sys
import tempfile
import unittest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from clean_model.data import split_by_group
from clean_model.qwen_lora import (
    _select_test_examples,
    encode_assistant_turn,
    prepare_splits,
    train_adapter,
)


class _FakeTokenizer:
    chat_template = 'fake-chat-template'

    def __init__(self, merge_assistant_boundary=False):
        self.merge_assistant_boundary = merge_assistant_boundary

    def apply_chat_template(
        self,
        messages,
        tokenize,
        add_generation_prompt,
        enable_thinking,
    ):
        if tokenize:
            raise AssertionError(
                'chat-template and rendered-text tokenization paths differ')
        assert not enable_thinking
        rendered = ''
        for message in messages:
            rendered += (
                '<|im_start|>%s\n%s<|im_end|>\n'
                % (message['role'], message['content']))
        if add_generation_prompt:
            rendered += '<|im_start|>assistant\n'
        return rendered

    def __call__(
        self,
        text,
        add_special_tokens,
        return_offsets_mapping,
    ):
        assert not add_special_tokens
        input_ids = [ord(char) for char in text]
        offsets = [(index, index + 1) for index in range(len(text))]
        marker = '<|im_start|>assistant\n'
        boundary = text.rfind(marker) + len(marker)
        if self.merge_assistant_boundary and 0 < boundary < len(text):
            input_ids[boundary - 1] = 1000000 + ord(text[boundary])
            del input_ids[boundary]
            offsets[boundary - 1] = (boundary - 1, boundary + 1)
            del offsets[boundary]
        result = {'input_ids': input_ids}
        if return_offsets_mapping:
            result['offset_mapping'] = offsets
        return result


class TestQwenLora(unittest.TestCase):
    def test_adapter_model_is_moved_to_cuda_before_training(self):
        source_path = os.path.join(BASE, 'clean_model', 'qwen_lora.py')
        with open(source_path, encoding='utf-8') as source_file:
            source = source_file.read()
        self.assertIn("    ).to('cuda')", source)

    def test_notebook_removes_unused_optional_torchao(self):
        notebook_path = os.path.join(
            BASE, 'kaggle_qwen3_0_6b_lora.ipynb')
        with open(notebook_path, encoding='utf-8') as notebook_file:
            notebook = json.load(notebook_file)
        setup_source = ''.join(notebook['cells'][1]['source'])
        self.assertIn("'pip', 'uninstall', '-y', 'torchao'", setup_source)
        self.assertIn('This pilot does not use torchao', setup_source)

    def test_combined_notebook_downloads_inputs_and_verifies_private_data(self):
        notebook_path = os.path.join(
            BASE, 'kaggle_qwen3_0_6b_lora_expanded.ipynb')
        with open(notebook_path, encoding='utf-8') as notebook_file:
            notebook = json.load(notebook_file)
        setup_source = ''.join(notebook['cells'][1]['source'])
        training_source = ''.join(notebook['cells'][3]['source'])
        self.assertIn(
            "GITHUB_REF = 'qwen3-combined-private-001'", setup_source)
        self.assertIn('git\', \'clone\'', setup_source)
        self.assertIn(
            "rglob(private_archive_name)", setup_source)
        self.assertIn(
            "private_archive_name = 'qwen3_private_dialogue_addon_001.zip'",
            setup_source)
        self.assertIn("bundle_manifest['review_status'] != 'draft'", setup_source)
        self.assertIn("'CC BY 4.0'", setup_source)
        self.assertIn('require_approved=False', training_source)
        self.assertIn('records = public_records + private_records', setup_source)
        self.assertIn('private_groups <= train_groups', setup_source)
        self.assertNotIn('Masamın üstü çok dağıldı', ''.join(
            line for cell in notebook['cells']
            for line in cell.get('source', [])))

    def test_combined_notebook_records_gguf_and_builds_one_download(self):
        notebook_path = os.path.join(
            BASE, 'kaggle_qwen3_0_6b_lora_expanded.ipynb')
        with open(notebook_path, encoding='utf-8') as notebook_file:
            notebook = json.load(notebook_file)
        export_source = ''.join(notebook['cells'][7]['source'])
        self.assertIn('actual_llama_cpp_commit != llama_cpp_commit',
                      export_source)
        self.assertIn('adapter_gguf_path.stat().st_size <= 1024',
                      export_source)
        self.assertIn("training_metadata['gguf_adapter']", export_source)
        self.assertIn('qwen3_0_6b_lora_run_003.zip', export_source)
        self.assertIn('required_deliverables', export_source)
        self.assertIn('archive.testzip()', export_source)
        self.assertIn('missing_from_archive', export_source)

    def test_loss_mask_only_trains_assistant_completion(self):
        result = encode_assistant_turn(
            [{'role': 'user', 'content': 'Selam'}],
            'Merhaba',
            _FakeTokenizer(),
        )
        first_target = len(
            '<|im_start|>user\nSelam<|im_end|>\n<|im_start|>assistant\n')
        self.assertEqual(result['labels'][:first_target], [-100] * first_target)
        self.assertEqual(
            result['labels'][first_target:],
            result['input_ids'][first_target:],
        )
        self.assertIn(ord('M'), result['labels'][first_target:])
        self.assertIn(ord('<'), result['labels'][first_target:])

    def test_rejects_context_without_final_user_turn(self):
        with self.assertRaisesRegex(ValueError, 'end with a user message'):
            encode_assistant_turn(
                [{'role': 'assistant', 'content': 'Merhaba'}],
                'Nasılsın?',
                _FakeTokenizer(),
            )

    def test_masks_token_that_crosses_prompt_response_boundary(self):
        tokenizer = _FakeTokenizer(merge_assistant_boundary=True)
        context = [{'role': 'user', 'content': 'Selam'}]
        result = encode_assistant_turn(context, 'Merhaba', tokenizer)
        full_text = tokenizer.apply_chat_template(
            [*context, {'role': 'assistant', 'content': 'Merhaba'}],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        offsets = tokenizer(
            full_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )['offset_mapping']
        merged_index = next(
            index for index, offset in enumerate(offsets)
            if offset[1] - offset[0] == 2
        )
        self.assertEqual(result['labels'][merged_index], -100)
        self.assertIn(ord('e'), result['labels'])

    def test_refuses_nonempty_training_output_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = os.path.join(directory, 'existing.txt')
            with open(marker, 'w', encoding='utf-8') as output:
                output.write('preserve')
            with self.assertRaisesRegex(
                    FileExistsError, 'training output directory is not empty'):
                train_adapter('unused.jsonl', directory)
            with open(marker, encoding='utf-8') as output:
                self.assertEqual(output.read(), 'preserve')

    def test_approved_example_records_have_isolated_group_splits(self):
        records = [{
            'schema_version': 1,
            'id': 'approved-%d' % index,
            'group_id': 'family-%d' % index,
            'kind': 'dialogue',
            'messages': [
                {'role': 'user', 'content': 'Question %d?' % index},
                {'role': 'assistant', 'content': 'Answer %d.' % index},
            ],
            'evidence': [],
            'provenance': {
                'dataset': 'Synthetic test fixture',
                'license': 'Test-only',
                'reference': 'Unit-test fixture, not source data.',
            },
            'review': {
                'status': 'approved',
                'reviewed_by': 'unit_test',
                'reviewed_at': '2026-10-15',
            },
        } for index in range(6)]
        splits = split_by_group(records, seed=7)
        groups = {
            name: {record['group_id'] for record in items}
            for name, items in splits.items()
        }
        self.assertFalse(groups['train'] & groups['validation'])
        self.assertFalse(groups['train'] & groups['test'])
        self.assertFalse(groups['validation'] & groups['test'])

    def test_draft_dataset_requires_explicit_opt_in_for_qwen_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'drafts.jsonl')
            with open(path, 'w', encoding='utf-8') as output:
                for index in range(6):
                    output.write(json.dumps({
                        'schema_version': 1,
                        'id': 'draft-%d' % index,
                        'group_id': 'family-%d' % index,
                        'kind': 'dialogue',
                        'messages': [
                            {'role': 'user', 'content': 'Soru %d?' % index},
                            {'role': 'assistant', 'content': 'Yanıt %d.' % index},
                        ],
                        'evidence': [],
                        'provenance': {
                            'dataset': 'test',
                            'license': 'CC BY 4.0',
                            'reference': 'https://example.invalid',
                        },
                        'review': {'status': 'draft'},
                    }, ensure_ascii=False) + '\n')

            with self.assertRaisesRegex(ValueError, 'status=approved'):
                prepare_splits(path, tokenizer=_FakeTokenizer())

            splits, summary = prepare_splits(
                path,
                tokenizer=_FakeTokenizer(),
                require_approved=False,
            )

        self.assertEqual(summary['review_statuses'], ['draft'])
        self.assertEqual(
            summary['record_counts'],
            {'train': 4, 'validation': 1, 'test': 1},
        )
        groups = {
            name: {row['group_id'] for row in rows}
            for name, rows in splits.items()
        }
        self.assertFalse(groups['train'] & groups['validation'])
        self.assertFalse(groups['train'] & groups['test'])
        self.assertFalse(groups['validation'] & groups['test'])

    def test_test_turn_sample_is_bounded_and_reproducible(self):
        examples = [{'record_id': 'record-%d' % index} for index in range(100)]
        first = _select_test_examples(examples, seed=7, max_test_examples=12)
        second = _select_test_examples(examples, seed=7, max_test_examples=12)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 12)
        indices = [
            int(item['record_id'].rsplit('-', 1)[1])
            for item in first
        ]
        self.assertEqual(indices, sorted(indices))

    def test_test_turn_sample_requires_positive_limit(self):
        with self.assertRaisesRegex(ValueError, 'must be positive'):
            _select_test_examples([], seed=7, max_test_examples=0)


if __name__ == '__main__':
    unittest.main()
