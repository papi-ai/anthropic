<?php

/*
 * This file is part of PapiAI,
 * A simple but powerful PHP library for building AI agents.
 *
 * (c) Marcello Duarte <marcello.duarte@gmail.com>
 *
 * For the full copyright and license information, please view the LICENSE
 * file that was distributed with this source code.
 */

declare(strict_types=1);

use PapiAI\Anthropic\AnthropicProvider;
use PapiAI\Anthropic\OutputSchema;
use PapiAI\Core\Message;
use PapiAI\Core\Schema\Schema;

/**
 * Captures the request payload so the output format can be asserted without HTTP.
 */
class TestableAnthropicStructuredOutputProvider extends AnthropicProvider
{
    public array $lastPayload = [];

    protected function request(array $payload): array
    {
        $this->lastPayload = $payload;

        return ['content' => [['type' => 'text', 'text' => '{"approved": true}']], 'stop_reason' => 'end_turn'];
    }
}

describe('AnthropicProvider structured output', function () {
    beforeEach(function () {
        $this->provider = new TestableAnthropicStructuredOutputProvider('test-api-key', AnthropicProvider::MODEL_CLAUDE_SONNET_5);
        $this->schema = Schema::object(['approved' => Schema::boolean()])->toJsonSchema();
    });

    it('enforces the schema itself', function () {
        expect($this->provider->supportsStructuredOutput())->toBeTrue();
    });

    it('sends the schema as the output format', function () {
        $this->provider->chat([Message::user('Review this.')], ['outputSchema' => $this->schema]);

        expect($this->provider->lastPayload['output_config'])->toBe([
            'format' => ['type' => 'json_schema', 'schema' => $this->schema],
        ]);
    });

    it('keeps the effort level beside the output format', function () {
        $this->provider->chat([Message::user('Review this.')], ['outputSchema' => $this->schema, 'effort' => 'low']);

        expect($this->provider->lastPayload['output_config']['effort'])->toBe('low');
        expect($this->provider->lastPayload['output_config']['format']['type'])->toBe('json_schema');
    });

    it('sends no output format without a schema', function () {
        $this->provider->chat([Message::user('Hello')]);

        expect($this->provider->lastPayload)->not->toHaveKey('output_config');
    });

    it('returns the JSON as the response text', function () {
        $response = $this->provider->chat([Message::user('Review this.')], ['outputSchema' => $this->schema]);

        expect($response->text)->toBe('{"approved": true}');
    });
});

describe('OutputSchema', function () {
    it('moves the constraints the API rejects into the description', function () {
        $schema = Schema::object([
            'age' => Schema::integer()->min(0)->max(150)->description('Age in years'),
            'tags' => Schema::array(Schema::string())->minItems(2)->maxItems(5),
        ])->toJsonSchema();

        $sent = (new OutputSchema())->forApi($schema);

        expect($sent['properties']['age'])->toBe([
            'type' => 'integer',
            'description' => 'Age in years (minimum: 0, maximum: 150)',
        ]);
        expect($sent['properties']['tags'])->toBe([
            'type' => 'array',
            'items' => ['type' => 'string'],
            'description' => '(minItems: 2, maxItems: 5)',
        ]);
    });

    it('keeps what the API accepts', function () {
        $schema = [
            'type' => 'object',
            'properties' => [
                'note' => ['type' => ['string', 'null']],
                'code' => ['type' => 'string', 'pattern' => '[A-Z]{3}'],
                'when' => ['type' => 'string', 'format' => 'date-time'],
                'items' => ['type' => 'array', 'items' => ['type' => 'string'], 'minItems' => 1],
            ],
            'required' => ['note', 'code', 'when', 'items'],
            'additionalProperties' => false,
        ];

        expect((new OutputSchema())->forApi($schema))->toBe($schema);
    });

    it('closes every object, however deep', function () {
        $schema = [
            'type' => 'object',
            'properties' => [
                'author' => ['type' => 'object', 'properties' => ['name' => ['type' => 'string']]],
                'reviews' => ['type' => 'array', 'items' => ['type' => 'object', 'properties' => [], 'additionalProperties' => true]],
                'choice' => ['anyOf' => [['type' => 'object', 'properties' => []], ['type' => 'null']]],
            ],
            '$defs' => ['point' => ['type' => 'object', 'properties' => ['x' => ['type' => 'number', 'minimum' => 0]]]],
        ];

        $sent = (new OutputSchema())->forApi($schema);

        expect($sent['additionalProperties'])->toBeFalse();
        expect($sent['properties']['author']['additionalProperties'])->toBeFalse();
        expect($sent['properties']['reviews']['items']['additionalProperties'])->toBeFalse();
        expect($sent['properties']['choice']['anyOf'][0]['additionalProperties'])->toBeFalse();
        expect($sent['$defs']['point']['additionalProperties'])->toBeFalse();
        expect($sent['$defs']['point']['properties']['x'])->toBe(['type' => 'number', 'description' => '(minimum: 0)']);
    });

    it('spells a nullable enum as a choice, since the API takes an enum under one type only', function () {
        $schema = Schema::enum(['open', 'closed'])->nullable()->description('Ticket state')->toJsonSchema();

        expect((new OutputSchema())->forApi($schema))->toBe([
            'description' => 'Ticket state',
            'anyOf' => [
                ['type' => 'string', 'enum' => ['open', 'closed']],
                ['type' => 'null'],
            ],
        ]);
    });

    it('drops a string format the API does not know', function () {
        $sent = (new OutputSchema())->forApi(['type' => 'string', 'format' => 'credit-card']);

        expect($sent)->toBe(['type' => 'string', 'description' => '(format: credit-card)']);
    });

    it('leaves a property that happens to be named like a keyword', function () {
        $schema = [
            'type' => 'object',
            'properties' => ['minimum' => ['type' => 'number'], 'format' => ['type' => 'string']],
            'additionalProperties' => false,
        ];

        expect((new OutputSchema())->forApi($schema))->toBe($schema);
    });
});
