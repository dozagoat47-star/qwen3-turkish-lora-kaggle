import unittest

from clean_model.qwen_lora import encode_assistant_turn


class FakeTokenizer:
    chat_template = 'test'

    def apply_chat_template(
        self,
        messages,
        tokenize,
        add_generation_prompt,
        enable_thinking,
    ):
        if not tokenize or enable_thinking:
            raise AssertionError('unexpected chat-template options')
        tokens = []
        for message in messages:
            tokens.extend([
                {'user': 10, 'assistant': 20}[message['role']],
                *message['content'].encode('utf-8'),
                30,
            ])
        if add_generation_prompt:
            tokens.append(20)
        else:
            tokens.append(50)
        return tokens


class TestQwenLoraEncoding(unittest.TestCase):
    def test_masks_prompt_and_trains_assistant_completion(self):
        result = encode_assistant_turn(
            [{'role': 'user', 'content': 'Selam'}],
            'Merhaba',
            FakeTokenizer(),
        )
        prefix_length = len([10, *'Selam'.encode('utf-8'), 30, 20])
        self.assertEqual(result['labels'][:prefix_length], [-100] * prefix_length)
        self.assertEqual(
            result['labels'][prefix_length:],
            result['input_ids'][prefix_length:],
        )

    def test_requires_user_message_at_end_of_context(self):
        with self.assertRaisesRegex(ValueError, 'end with a user message'):
            encode_assistant_turn(
                [{'role': 'assistant', 'content': 'Hello'}],
                'Hi',
                FakeTokenizer(),
            )


if __name__ == '__main__':
    unittest.main()
