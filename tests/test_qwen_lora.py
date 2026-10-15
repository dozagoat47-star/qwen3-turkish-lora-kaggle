import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from clean_model.qwen_lora import (
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
    def test_expanded_notebook_downloads_pinned_bundle_without_kaggle_input(self):
        root = Path(__file__).resolve().parents[1]
        notebook = json.loads(
            (root / 'kaggle_qwen3_0_6b_lora_expanded.ipynb').read_text(
                encoding='utf-8'))
        setup = ''.join(notebook['cells'][1]['source'])
        self.assertIn("GITHUB_REF = 'qwen3-expanded-draft-001'", setup)
        self.assertIn("'git', 'clone'", setup)
        self.assertIn("datasets/qwen3_0_6b_lora_bundle_002.zip", setup)
        self.assertNotIn("Path('/kaggle/input')", setup)

    def test_published_bundle_is_hash_checked_and_attributed(self):
        root = Path(__file__).resolve().parents[1]
        bundle_path = root / 'datasets/qwen3_0_6b_lora_bundle_002.zip'
        with ZipFile(bundle_path) as bundle:
            manifest = json.loads(
                bundle.read('bundle_manifest.json').decode('utf-8'))
            self.assertEqual(manifest['record_count'], 2480)
            self.assertEqual(manifest['review_status'], 'draft')
            self.assertEqual(manifest['license'], 'CC BY 4.0')
            self.assertEqual(
                manifest['source_selection']['source_split'], 'train only')
            self.assertIn(
                'data/raw/turkish_daily_dialogues_sft_001/LICENSE',
                bundle.namelist())
            for name, expected_hash in manifest['files'].items():
                self.assertEqual(
                    hashlib.sha256(bundle.read(name)).hexdigest(),
                    expected_hash,
                )

    def test_draft_records_need_explicit_opt_in_for_qwen_splits(self):
        records = [{
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
        } for index in range(6)]
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / 'drafts.jsonl'
            data_path.write_text(
                ''.join(json.dumps(record) + '\n' for record in records),
                encoding='utf-8',
            )
            with self.assertRaisesRegex(ValueError, 'status=approved'):
                prepare_splits(data_path, tokenizer=_FakeTokenizer())
            splits, summary = prepare_splits(
                data_path,
                tokenizer=_FakeTokenizer(),
                require_approved=False,
            )

        self.assertEqual(summary['review_statuses'], ['draft'])
        self.assertTrue(splits['train'])
        self.assertTrue(splits['validation'])
        self.assertTrue(splits['test'])

    def test_adapter_model_is_moved_to_cuda_before_training(self):
        source_path = Path(__file__).resolve().parents[1]
        source = (source_path / 'clean_model' / 'qwen_lora.py').read_text(
            encoding='utf-8')
        self.assertIn("    ).to('cuda')", source)

    def test_notebook_removes_unused_optional_torchao(self):
        notebook_path = Path(__file__).resolve().parents[1]
        notebook = json.loads(
            (notebook_path / 'kaggle_qwen3_0_6b_lora.ipynb').read_text(
                encoding='utf-8'))
        setup_source = ''.join(notebook['cells'][1]['source'])
        self.assertIn("'pip', 'uninstall', '-y', 'torchao'", setup_source)
        self.assertIn('This pilot does not use torchao', setup_source)

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
            marker = Path(directory) / 'existing.txt'
            marker.write_text('preserve', encoding='utf-8')
            with self.assertRaisesRegex(
                    FileExistsError, 'training output directory is not empty'):
                train_adapter('unused.jsonl', directory)
            self.assertEqual(marker.read_text(encoding='utf-8'), 'preserve')


if __name__ == '__main__':
    unittest.main()
