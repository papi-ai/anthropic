# PapiAI Anthropic Provider

[![Tests](https://github.com/papi-ai/anthropic/workflows/Tests/badge.svg)](https://github.com/papi-ai/anthropic/actions?query=workflow%3ATests)

Anthropic Claude provider for [PapiAI](https://github.com/papi-ai/papi-core) - A simple but powerful PHP library for building AI agents.

## Installation

```bash
composer require papi-ai/anthropic
```

## Usage

```php
use PapiAI\Core\Agent;
use PapiAI\Anthropic\AnthropicProvider;

$provider = new AnthropicProvider(
    apiKey: $_ENV['ANTHROPIC_API_KEY'],
);

$agent = new Agent(
    provider: $provider,
    model: 'claude-sonnet-4-20250514',
    instructions: 'You are a helpful assistant.',
);

$response = $agent->run('Hello!');
echo $response->text;
```

## Available Models

- `claude-sonnet-4-20250514` (default)
- `claude-3-opus-20240229`
- `claude-3-sonnet-20240229`
- `claude-3-haiku-20240307`

## Features

- Tool/function calling
- Vision/multimodal support
- Streaming support

## License

MIT
