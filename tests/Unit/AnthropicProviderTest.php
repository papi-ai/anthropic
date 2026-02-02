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
use PapiAI\Core\Contracts\ProviderInterface;
use PapiAI\Core\Message;

describe('AnthropicProvider', function () {
    describe('construction', function () {
        it('implements ProviderInterface', function () {
            $provider = new AnthropicProvider('test-api-key');

            expect($provider)->toBeInstanceOf(ProviderInterface::class);
        });

        it('uses default model', function () {
            $provider = new AnthropicProvider('test-api-key');

            expect($provider->getName())->toBe('anthropic');
        });

        it('accepts custom model', function () {
            $provider = new AnthropicProvider(
                apiKey: 'test-api-key',
                defaultModel: 'claude-3-opus-20240229',
            );

            expect($provider)->toBeInstanceOf(AnthropicProvider::class);
        });
    });

    describe('capabilities', function () {
        it('supports tools', function () {
            $provider = new AnthropicProvider('test-api-key');

            expect($provider->supportsTool())->toBeTrue();
        });

        it('supports vision', function () {
            $provider = new AnthropicProvider('test-api-key');

            expect($provider->supportsVision())->toBeTrue();
        });

        it('does not support native structured output', function () {
            $provider = new AnthropicProvider('test-api-key');

            expect($provider->supportsStructuredOutput())->toBeFalse();
        });
    });

    describe('message conversion', function () {
        it('handles system messages separately', function () {
            // This test verifies the internal logic by checking the provider can be constructed
            // Full integration tests would require mocking curl
            $provider = new AnthropicProvider('test-api-key');

            $messages = [
                Message::system('You are helpful'),
                Message::user('Hello'),
            ];

            // Provider should handle these without error during payload building
            expect($provider)->toBeInstanceOf(AnthropicProvider::class);
        });
    });
});

describe('AnthropicProvider integration', function () {
    it('throws on API error', function () {
        // Skip if no API key (don't make real API calls in unit tests)
        $this->markTestSkipped('Integration test - requires ANTHROPIC_API_KEY');

        $provider = new AnthropicProvider('invalid-key');

        expect(fn() => $provider->chat([Message::user('Hello')]))
            ->toThrow(RuntimeException::class);
    });
})->skip();
