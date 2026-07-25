# Anthropic (Claude)

Anthropic Claude provider for PapiAI.

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

## Models

- `claude-sonnet-4-20250514` (default)
- `claude-3-opus-20240229`
- `claude-3-sonnet-20240229`
- `claude-3-haiku-20240307`

## Capabilities

| Capability | Supported |
|---|---|
| Chat | Yes |
| Streaming | Yes |
| Tool calling | Yes |
| Vision | Yes |
| Structured output | Yes |
| Prompt caching | Yes |

## Requirements

- PHP 8.2+
- `ext-curl`
- `papi-ai/papi-core` ^0.12
