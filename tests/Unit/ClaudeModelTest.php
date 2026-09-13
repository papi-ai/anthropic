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
use PapiAI\Anthropic\ClaudeModel;
use PapiAI\Anthropic\ThinkingApi;
use PapiAI\Core\Effort;

describe('ClaudeModel', function () {
    it('is the source of truth the old constants alias, so nothing existing breaks', function () {
        expect(AnthropicProvider::MODEL_CLAUDE_SONNET_5)->toBe(ClaudeModel::Sonnet5->value);
        expect(AnthropicProvider::MODEL_CLAUDE_FABLE_5_1)->toBe(ClaudeModel::Fable51->value);
        expect(AnthropicProvider::MODEL_CLAUDE_HAIKU_4_5)->toBe(ClaudeModel::Haiku45->value);
        expect(ClaudeModel::from('claude-opus-4-6'))->toBe(ClaudeModel::Opus46);
    });

    it('knows every ID it ships, so a watchdog can enumerate instead of parsing source', function () {
        $ids = array_map(fn (ClaudeModel $m) => $m->value, ClaudeModel::cases());

        expect($ids)->toContain('claude-fable-5-1', 'claude-opus-5', 'claude-sonnet-5', 'claude-haiku-4-5');
        expect($ids)->toBe(array_unique($ids));
    });

    it('returns null for an ID it has not heard of, rather than throwing', function () {
        // Callers pass next month's model before we ship a case for it. That must keep working.
        expect(ClaudeModel::tryFrom('claude-opus-6'))->toBeNull();
    });

    describe('retirement', function () {
        it('carries no deprecation today, since every shipped model is active', function () {
            foreach (ClaudeModel::cases() as $model) {
                expect($model->isDeprecated())->toBeFalse();
            }
        });

        it('knows the one retirement floor inside a year', function () {
            expect(ClaudeModel::Haiku45->retiresNotBefore())->toBe('2026-10-15');
            expect(ClaudeModel::Sonnet5->retiresNotBefore())->toBe('2027-06-30');
        });
    });

    describe('thinking API', function () {
        it('is a token budget on Haiku 4.5, the last pre-adaptive model', function () {
            expect(ClaudeModel::Haiku45->thinking())->toBe(ThinkingApi::Budget);
        });

        it('is adaptive from 4.6 onwards', function () {
            foreach ([ClaudeModel::Opus46, ClaudeModel::Sonnet46, ClaudeModel::Opus47, ClaudeModel::Opus48, ClaudeModel::Opus5, ClaudeModel::Sonnet5] as $model) {
                expect($model->thinking())->toBe(ThinkingApi::Adaptive);
            }
        });

        it('is always on for Fable, which cannot be told to stop', function () {
            expect(ClaudeModel::Fable5->thinking())->toBe(ThinkingApi::AlwaysOn);
            expect(ClaudeModel::Fable51->thinking())->toBe(ThinkingApi::AlwaysOn);
        });

        it('says which adaptive models think when the field is omitted', function () {
            expect(ClaudeModel::Sonnet5->thinksByDefault())->toBeTrue();
            expect(ClaudeModel::Opus5->thinksByDefault())->toBeTrue();
            expect(ClaudeModel::Opus48->thinksByDefault())->toBeFalse();
            expect(ClaudeModel::Sonnet46->thinksByDefault())->toBeFalse();
        });
    });

    describe('effort levels', function () {
        it('lacks xhigh on the 4.6 pair', function () {
            expect(ClaudeModel::Sonnet46->effortLevels())->toBe([Effort::Low, Effort::Medium, Effort::High, Effort::Maximum]);
        });

        it('offers the full five from 4.7 onwards', function () {
            expect(ClaudeModel::Opus5->effortLevels())
                ->toBe([Effort::Low, Effort::Medium, Effort::High, Effort::ExtraHigh, Effort::Maximum]);
        });

        it('offers none on Haiku 4.5, which rejects the parameter', function () {
            expect(ClaudeModel::Haiku45->effortLevels())->toBe([]);
        });
    });

    describe('forced tool choice', function () {
        it('is refused by Fable 5.1 alone', function () {
            expect(ClaudeModel::Fable51->acceptsForcedTools())->toBeFalse();
            expect(ClaudeModel::Fable5->acceptsForcedTools())->toBeTrue();
            expect(ClaudeModel::Sonnet5->acceptsForcedTools())->toBeTrue();
        });
    });
});
