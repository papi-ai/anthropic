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

/**
 * A JSON Schema narrowed to what Claude's structured outputs compile.
 *
 * The API answers 400 to numeric and length limits, to array sizes beyond a minimum of one, to
 * string formats outside its list and to an object left open. Those limits move into the
 * description, where the model still reads them, and every object is closed. The caller's own
 * schema keeps its limits, so a reply that breaks one is still caught on return.
 *
 * @see https://platform.claude.com/docs/en/build-with-claude/structured-outputs#json-schema-limitations
 */
final class OutputSchema
{
    private const REJECTED = [
        'minimum', 'maximum', 'exclusiveMinimum', 'exclusiveMaximum', 'multipleOf',
        'minLength', 'maxLength', 'maxItems', 'uniqueItems',
    ];

    private const FORMATS = [
        'date-time', 'time', 'date', 'duration', 'email', 'hostname', 'uri', 'ipv4', 'ipv6', 'uuid',
    ];

    private const SCHEMA_MAPS = ['properties', '$defs', 'definitions'];

    private const SCHEMA_LISTS = ['anyOf', 'allOf'];

    /**
     * @param array<string, mixed> $schema A JSON Schema
     *
     * @return array<string, mixed> The same schema, as the API accepts it
     */
    public function forApi(array $schema): array
    {
        $schema = $this->withLimitsDescribed($this->withEnumUnderOneType($schema));

        foreach (self::SCHEMA_MAPS as $keyword) {
            if (is_array($schema[$keyword] ?? null)) {
                $schema[$keyword] = array_map($this->nested(...), $schema[$keyword]);
            }
        }

        foreach (self::SCHEMA_LISTS as $keyword) {
            if (is_array($schema[$keyword] ?? null)) {
                $schema[$keyword] = array_map($this->nested(...), $schema[$keyword]);
            }
        }

        if (is_array($schema['items'] ?? null)) {
            $schema['items'] = $this->forApi($schema['items']);
        }

        if ($this->isObject($schema)) {
            $schema['additionalProperties'] = false;
        }

        return $schema;
    }

    private function nested(mixed $schema): mixed
    {
        return is_array($schema) ? $this->forApi($schema) : $schema;
    }

    /**
     * An enum declared under several types, as a nullable one is, becomes a choice between them.
     *
     * @param array<string, mixed> $schema
     *
     * @return array<string, mixed>
     */
    private function withEnumUnderOneType(array $schema): array
    {
        $types = $schema['type'] ?? null;
        $values = $schema['enum'] ?? null;

        if (!is_array($types) || !is_array($values)) {
            return $schema;
        }

        unset($schema['type'], $schema['enum']);
        $named = array_values(array_filter($values, static fn (mixed $value): bool => $value !== null));
        $schema['anyOf'] = array_map(
            static fn (mixed $type): array => $type === 'null' ? ['type' => 'null'] : ['type' => $type, 'enum' => $named],
            array_values($types),
        );

        return $schema;
    }

    /**
     * @param array<string, mixed> $schema
     *
     * @return array<string, mixed>
     */
    private function withLimitsDescribed(array $schema): array
    {
        $moved = [];

        foreach ($schema as $keyword => $value) {
            if ($this->isRejected($keyword, $value)) {
                $moved[] = sprintf('%s: %s', $keyword, is_scalar($value) ? (string) $value : json_encode($value));
                unset($schema[$keyword]);
            }
        }

        if ($moved === []) {
            return $schema;
        }

        $limits = sprintf('(%s)', implode(', ', $moved));
        $schema['description'] = is_string($schema['description'] ?? null) ? $schema['description'].' '.$limits : $limits;

        return $schema;
    }

    private function isRejected(string $keyword, mixed $value): bool
    {
        return match (true) {
            in_array($keyword, self::REJECTED, true) => true,
            $keyword === 'minItems' => $value > 1,
            $keyword === 'format' => !in_array($value, self::FORMATS, true),
            default => false,
        };
    }

    /**
     * @param array<string, mixed> $schema
     */
    private function isObject(array $schema): bool
    {
        $type = $schema['type'] ?? null;

        return $type === 'object' || (is_array($type) && in_array('object', $type, true));
    }
}
