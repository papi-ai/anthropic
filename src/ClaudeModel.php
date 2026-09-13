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
 * The model, not the provider, decides how effort is spelt, whether thinking can be switched off
 * and whether a tool can be forced. Putting those answers here keeps the provider free of
 * string-sniffing, and lets a watchdog enumerate what we ship instead of parsing source.
 *
 * An ID we have not heard of is not an error: `tryFrom()` returns null and the provider assumes
 * the newest generation, because next month's model is far likelier than last year's.
 *
 * Retirement floors are Anthropic's published "not sooner than" dates, ISO formatted.
 *
 * @see https://platform.claude.com/docs/en/about-claude/model-deprecations
 */
enum ClaudeModel: string
{
    case Fable51 = 'claude-fable-5-1';
    case Fable5 = 'claude-fable-5';
    case Opus5 = 'claude-opus-5';
    case Opus48 = 'claude-opus-4-8';
    case Opus47 = 'claude-opus-4-7';
    case Opus46 = 'claude-opus-4-6';
    case Sonnet5 = 'claude-sonnet-5';
    case Sonnet46 = 'claude-sonnet-4-6';
    case Haiku45 = 'claude-haiku-4-5';

    /**
     * The shape this model accepts for thinking.
     */
    public function thinking(): ThinkingApi
    {
        return match ($this) {
            self::Fable51, self::Fable5 => ThinkingApi::AlwaysOn,
            self::Haiku45 => ThinkingApi::Budget,
            default => ThinkingApi::Adaptive,
        };
    }

    /**
     * Whether an adaptive model thinks when the request omits the thinking field.
     *
     * Sonnet 5 and Opus 5 do, so "none" has to say `disabled` explicitly. The 4.x models are off
     * when the field is omitted.
     */
    public function thinksByDefault(): bool
    {
        return match ($this) {
            self::Opus5, self::Sonnet5 => true,
            default => false,
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
     * Fable 5.1 returns 400 for both; "auto" and "none" still work there.
     */
    public function acceptsForcedTools(): bool
    {
        return $this !== self::Fable51;
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
     * A floor, not a schedule: "not sooner than". A model that has passed its floor is not
     * necessarily gone, but a default resting on one deserves a look.
     */
    public function retiresNotBefore(): string
    {
        return match ($this) {
            self::Fable51 => '2027-09-01',
            self::Fable5 => '2027-06-09',
            self::Opus5 => '2027-07-24',
            self::Opus48 => '2027-05-28',
            self::Opus47 => '2027-04-16',
            self::Opus46 => '2027-02-05',
            self::Sonnet5 => '2027-06-30',
            self::Sonnet46 => '2027-02-17',
            self::Haiku45 => '2026-10-15',
        };
    }
}
