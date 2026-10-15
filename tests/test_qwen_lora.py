import json
import os
import tempfile
import unittest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from clean_model.qwen_lora import encode_assistant_turn, train_adapter


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
    def test_notebook_removes_unused_optional_torchao(self):
        notebook_path = os.path.join(
            BASE, 'kaggle_qwen3_0_6b_lora.ipynb')
        with open(notebook_path, encoding='utf-8') as notebook_file:
            notebook = json.load(notebook_file)
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
            marker = os.path.join(directory, 'existing.txt')
            with open(marker, 'w', encoding='utf-8') as output:
                output.write('preserve')
            with self.assertRaisesRegex(
                    FileExistsError, 'training output directory is not empty'):
                train_adapter('unused.jsonl', directory)
            with open(marker, encoding='utf-8') as output:
                self.assertEqual(output.read(), 'preserve')

if __name__ == '__main__':
    unittest.main()
