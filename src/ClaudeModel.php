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

namespace PapiAI\Anthropic;

use PapiAI\Core\Effort;

/**
 * Every Claude model this package knows, and what each one accepts on the wire.
 *
 * The model, not the provider, decides how effort is spelt, whether and how thinking can be
 * switched off, whether sampling parameters or a forced tool choice are accepted. Putting those
 * answers here keeps the provider free of string-sniffing, and lets a watchdog enumerate what we
 * ship instead of parsing source.
 *
 * An ID we have not heard of is not an error: `tryFrom()` returns null and the provider assumes
 * the newest generation's rules.
 *
 * Retirement floors are Anthropic's published "not sooner than" dates, ISO formatted.
 *
 * @see https://platform.claude.com/docs/en/models/overview
 * @see https://platform.claude.com/docs/en/about-claude/model-deprecations
 */
enum ClaudeModel: string
{
    case Fable52 = 'claude-fable-5-2';
    case Fable51 = 'claude-fable-5-1';
    case Fable5 = 'claude-fable-5';
    case Mythos52 = 'claude-mythos-5-2';
    case Mythos51 = 'claude-mythos-5-1';
    case Mythos5 = 'claude-mythos-5';
    case Opus55 = 'claude-opus-5-5';
    case Opus5 = 'claude-opus-5';
    case Opus48 = 'claude-opus-4-8';
    case Opus47 = 'claude-opus-4-7';
    case Opus46 = 'claude-opus-4-6';
    case Sonnet55 = 'claude-sonnet-5-5';
    case Sonnet5 = 'claude-sonnet-5';
    case Sonnet46 = 'claude-sonnet-4-6';
    case Haiku55 = 'claude-haiku-5-5';
    case Haiku45 = 'claude-haiku-4-5';

    /**
     * The shape this model accepts for thinking.
     */
    public function thinking(): ThinkingApi
    {
        return match ($this) {
            self::Fable52, self::Fable51, self::Fable5,
            self::Mythos52, self::Mythos51, self::Mythos5,
            self::Opus55 => ThinkingApi::AlwaysOn,
            self::Haiku45 => ThinkingApi::Budget,
            default => ThinkingApi::Adaptive,
        };
    }

    /**
     * The `thinking.type` that turns thinking off, or null when the model offers none.
     *
     * Sonnet 5.5 rejects `disabled`; its floor is `between_tools`, which turns off up-front
     * thinking only. The always-on models have no switch at all.
     */
    public function offSwitch(): ?string
    {
        return match ($this) {
            self::Sonnet55 => 'between_tools',
            self::Haiku55, self::Opus5, self::Sonnet5,
            self::Opus48, self::Opus47, self::Opus46, self::Sonnet46 => 'disabled',
            default => null,
        };
    }

    /**
     * Whether an adaptive model thinks when the request omits the thinking field.
     *
     * From Opus 5 on they do, so "none" has to say so explicitly. The 4.x models are off when
     * the field is omitted.
     */
    public function thinksByDefault(): bool
    {
        return match ($this) {
            self::Opus48, self::Opus47, self::Opus46, self::Sonnet46, self::Haiku45 => false,
            default => true,
        };
    }

    /**
     * The effort the API applies when none is sent. Opus 5.5 and Haiku 5.5 default to medium.
     */
    public function defaultEffort(): Effort
    {
        return match ($this) {
            self::Opus55, self::Haiku55 => Effort::Medium,
            default => Effort::High,
        };
    }

    /**
     * The effort levels this model accepts, in the neutral vocabulary.
     *
     * Empty for Haiku 4.5, which rejects the parameter outright. The 4.6 pair predates `xhigh`.
     *
     * @return list<Effort>
     */
    public function effortLevels(): array
    {
        return match ($this) {
            self::Haiku45 => [],
            self::Opus46, self::Sonnet46 => [Effort::Low, Effort::Medium, Effort::High, Effort::Maximum],
            default => [Effort::Low, Effort::Medium, Effort::High, Effort::ExtraHigh, Effort::Maximum],
        };
    }

    /**
     * Whether a forced tool choice ("required" or a named tool) is accepted.
     *
     * Fable 5.1 and later, Mythos 5.1 and later, Opus 5.5 and Sonnet 5.5 return 400 for both;
     * "auto" and "none" still work everywhere.
     */
    public function acceptsForcedTools(): bool
    {
        return match ($this) {
            self::Fable52, self::Fable51, self::Mythos52, self::Mythos51, self::Opus55, self::Sonnet55 => false,
            default => true,
        };
    }

    /**
     * Whether `temperature`, `top_p` and `top_k` are accepted. Every model from 4.7 on rejects them.
     */
    public function acceptsSampling(): bool
    {
        return match ($this) {
            self::Opus46, self::Sonnet46, self::Haiku45 => true,
            default => false,
        };
    }

    /**
     * Whether a `role: system` message is accepted after the conversation has started.
     *
     * It lets an instruction change without editing the opening prompt, which would invalidate the
     * thinking and the cache that follow. Sonnet 5 and the models before Opus 4.8 reject it.
     */
    public function acceptsMidConversationSystem(): bool
    {
        return match ($this) {
            self::Sonnet5, self::Opus47, self::Opus46, self::Sonnet46, self::Haiku45 => false,
            default => true,
        };
    }

    /**
     * Whether Anthropic has announced a retirement for this model.
     */
    public function isDeprecated(): bool
    {
        return $this->retiredOn() !== null;
    }

    /**
     * The announced retirement date, ISO formatted, once one exists. Null while the model is active.
     */
    public function retiredOn(): ?string
    {
        return null;
    }

    /**
     * The earliest date Anthropic will retire this model, ISO formatted.
     *
     * A floor, not a schedule: "not sooner than".
     */
    public function retiresNotBefore(): string
    {
        return match ($this) {
            self::Fable52, self::Mythos52 => '2027-11-09',
            self::Fable51, self::Mythos51 => '2027-09-01',
            self::Fable5, self::Mythos5 => '2027-06-09',
            self::Opus55 => '2027-09-22',
            self::Opus5 => '2027-07-24',
            self::Opus48 => '2027-05-28',
            self::Opus47 => '2027-04-16',
            self::Opus46 => '2027-02-05',
            self::Sonnet55 => '2027-09-28',
            self::Sonnet5 => '2027-06-30',
            self::Sonnet46 => '2027-02-17',
            self::Haiku55 => '2027-10-07',
            self::Haiku45 => '2026-10-15',
        };
    }
}
