# PapiAI Anthropic Provider

[![CI](https://github.com/papi-ai/anthropic/workflows/CI/badge.svg)](https://github.com/papi-ai/anthropic/actions?query=workflow%3ACI) [![Latest Version](https://img.shields.io/packagist/v/papi-ai/anthropic.svg)](https://packagist.org/packages/papi-ai/anthropic) [![Total Downloads](https://img.shields.io/packagist/dt/papi-ai/anthropic.svg)](https://packagist.org/packages/papi-ai/anthropic) [![PHP Version](https://img.shields.io/packagist/php-v/papi-ai/anthropic.svg)](https://packagist.org/packages/papi-ai/anthropic) [![License](https://img.shields.io/packagist/l/papi-ai/anthropic.svg)](https://packagist.org/packages/papi-ai/anthropic)

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
    model: AnthropicProvider::MODEL_CLAUDE_SONNET_5,
    instructions: 'You are a helpful assistant.',
);

$response = $agent->run('Hello!');
echo $response->text;
```

## Available Models

```php
AnthropicProvider::MODEL_CLAUDE_FABLE_5_1  // 'claude-fable-5-1' (most capable)
AnthropicProvider::MODEL_CLAUDE_OPUS_5     // 'claude-opus-5'
AnthropicProvider::MODEL_CLAUDE_SONNET_5   // 'claude-sonnet-5' (default)
AnthropicProvider::MODEL_CLAUDE_HAIKU_4_5  // 'claude-haiku-4-5' (fastest)
AnthropicProvider::MODEL_CLAUDE_FABLE_5    // 'claude-fable-5'
AnthropicProvider::MODEL_CLAUDE_OPUS_4_8   // 'claude-opus-4-8'
AnthropicProvider::MODEL_CLAUDE_OPUS_4_7   // 'claude-opus-4-7'
AnthropicProvider::MODEL_CLAUDE_OPUS_4_6   // 'claude-opus-4-6'
AnthropicProvider::MODEL_CLAUDE_SONNET_4_6 // 'claude-sonnet-4-6'
```

All of these are active. Haiku 4.5 is the only one with a retirement floor inside a year: not sooner than 15 October 2026.

## Reasoning effort

`effort` maps to whichever shape the model's generation accepts. Everything from 4.6 on takes adaptive thinking plus an effort level (`low` to `max`); Haiku 4.5 takes a token budget carved out of `maxTokens`; Fable cannot stop thinking, so `none` narrows to `low` rather than sending a block the API rejects.

Fable 5.1 does not accept a forced tool choice. `toolChoice` of `required` or a named tool throws a `ProviderException` before any request is made; `auto` and `none` work as normal.

## Features

- Tool/function calling
- Vision/multimodal support
- Streaming support

## License

MIT
