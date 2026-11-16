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
    model: AnthropicProvider::MODEL_CLAUDE_SONNET_5_5,
    instructions: 'You are a helpful assistant.',
);

$response = $agent->run('Hello!');
echo $response->text;
```

## Available Models

```php
AnthropicProvider::MODEL_CLAUDE_FABLE_5_2  // 'claude-fable-5-2' (most capable)
AnthropicProvider::MODEL_CLAUDE_OPUS_5_5   // 'claude-opus-5-5'
AnthropicProvider::MODEL_CLAUDE_SONNET_5_5 // 'claude-sonnet-5-5' (default)
AnthropicProvider::MODEL_CLAUDE_HAIKU_5_5  // 'claude-haiku-5-5' (fastest)
AnthropicProvider::MODEL_CLAUDE_FABLE_5_1  // 'claude-fable-5-1'
AnthropicProvider::MODEL_CLAUDE_OPUS_5     // 'claude-opus-5'
AnthropicProvider::MODEL_CLAUDE_SONNET_5   // 'claude-sonnet-5'
AnthropicProvider::MODEL_CLAUDE_HAIKU_4_5  // 'claude-haiku-4-5'
AnthropicProvider::MODEL_CLAUDE_FABLE_5    // 'claude-fable-5'
AnthropicProvider::MODEL_CLAUDE_OPUS_4_8   // 'claude-opus-4-8'
AnthropicProvider::MODEL_CLAUDE_OPUS_4_7   // 'claude-opus-4-7'
AnthropicProvider::MODEL_CLAUDE_OPUS_4_6   // 'claude-opus-4-6'
AnthropicProvider::MODEL_CLAUDE_SONNET_4_6 // 'claude-sonnet-4-6'
```

All of these are active. Haiku 4.5 has passed its retirement floor (15 October 2026) and may go at any time.

The constants alias a `ClaudeModel` enum, which is the source of truth: it knows each model's thinking API, effort levels, whether a tool can be forced, and its retirement floor. Pass the enum's `->value` or the constant, both are the same string.

```php
use PapiAI\Anthropic\ClaudeModel;

ClaudeModel::Haiku45->retiresNotBefore();   // '2026-10-15'
ClaudeModel::Fable51->acceptsForcedTools(); // false
ClaudeModel::tryFrom('claude-opus-6');      // null: unknown IDs are assumed newest, never rejected
```

## Reasoning effort

`effort` maps to whichever shape the model's generation accepts. Everything from 4.6 on takes adaptive thinking plus an effort level (`low` to `max`); Haiku 4.5 takes a token budget carved out of `maxTokens`. Turning thinking off differs by model: `disabled` on most, `between_tools` on Sonnet 5.5, and not at all on Opus 5.5 or Fable, where `none` narrows to `low`.

Fable 5.1 and later, Opus 5.5 and Sonnet 5.5 do not accept a forced tool choice. `toolChoice` of `required` or a named tool throws a `ProviderException` before any request is made; `auto` and `none` work as normal.

Every model from 4.7 on rejects `temperature`, so passing one throws a `ProviderException` before any request is made.

## Features

- Tool/function calling
- Vision/multimodal support
- Streaming support

## Creating It by Name

An application that picks its provider from configuration can build this one without naming
its class:

```php
use PapiAI\Core\Factory\ComposerDiscovery;
use PapiAI\Core\Factory\ProviderRegistry;

$registry = new ProviderRegistry(new ComposerDiscovery());

$provider = $registry->create('anthropic', [
    'api_key' => $_ENV['ANTHROPIC_API_KEY'],
]);
```

It takes `api_key` (required), `model`, `max_tokens` and `effort`. Anything it is not given falls
back to the constructor's defaults, and a missing or unknown setting throws `ConfigurationException`
before any request is made.

## License

MIT
