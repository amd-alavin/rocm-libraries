/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once

#include <array>
#include <iostream>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace TensileLite
{
    namespace Fallback
    {
        struct OpenOpen
        {
            static std::string description()
            {
                return "open_open";
            }

            template <typename T>
            static bool contains(T value, T lower, T upper)
            {
                return value > lower && value < upper;
            }
        };

        template <typename KeyTag, typename Value>
        struct ValueBinding
        {
            using Key  = KeyTag;
            using Type = Value;

            Value value;
        };

        template <typename... Bindings>
        struct Context
        {
            using Tuple = std::tuple<Bindings...>;

            constexpr Context(Bindings... bindings)
                : values(bindings...)
            {
            }

            template <typename KeyTag>
            constexpr const typename KeyTag::Type& get() const
            {
                return std::get<ValueBinding<KeyTag, typename KeyTag::Type>>(values).value;
            }

            Tuple values;
        };

        template <typename KeyTag>
        constexpr ValueBinding<KeyTag, typename KeyTag::Type> bind(typename KeyTag::Type value)
        {
            return {value};
        }

        template <typename KeyTag, typename Value>
        struct Rule
        {
            using Key  = KeyTag;
            using Type = Value;
        };

        template <typename Value>
        struct RangeInterval
        {
            Value lower;
            Value upper;
        };

        template <typename KeyTag, typename Value>
        struct Category : Rule<KeyTag, Value>
        {
            Category() = default;

            explicit Category(std::vector<Value> allowedValues)
                : values(std::move(allowedValues))
            {
            }

            std::vector<Value> values;

            std::string description() const
            {
                return "Category";
            }

            bool valid(bool verbose = false) const
            {
                static_cast<void>(verbose);
                return true;
            }

            template <typename ContextT>
            bool matches(const ContextT& context) const
            {
                if(values.empty())
                    return true;

                const auto& actual = context.template get<KeyTag>();
                for(const auto& allowed : values)
                {
                    if(allowed == actual)
                        return true;
                }
                return false;
            }
        };

        template <typename KeyTag, typename Value, typename IntervalSemantics = OpenOpen>
        struct Range : Rule<KeyTag, Value>
        {
            using Interval = RangeInterval<Value>;

            Range() = default;

            explicit Range(std::vector<Interval> rangeIntervals)
                : intervals(std::move(rangeIntervals))
            {
            }

            std::vector<Interval> intervals;

            std::string description() const
            {
                return "Range";
            }

            bool valid(bool verbose = false) const
            {
                bool isValid = true;
                for(const auto& interval : intervals)
                {
                    if(!(interval.lower < interval.upper))
                    {
                        if(verbose)
                            std::cout << "Range: lower bound must be less than upper bound"
                                      << std::endl;
                        isValid = false;
                    }
                }
                return isValid;
            }

            template <typename ContextT>
            bool matches(const ContextT& context) const
            {
                if(intervals.empty())
                    return true;

                const auto& actual = context.template get<KeyTag>();
                for(const auto& interval : intervals)
                {
                    if(IntervalSemantics::contains(actual, interval.lower, interval.upper))
                        return true;
                }
                return false;
            }

            static std::vector<Interval> fromPairs(const std::vector<Value>& packed)
            {
                std::vector<Interval> intervals;
                intervals.reserve(packed.size() / 2);
                for(size_t i = 0; i + 1 < packed.size(); i += 2)
                {
                    intervals.push_back({packed[i], packed[i + 1]});
                }
                return intervals;
            }

            static bool validPairs(const std::vector<Value>& packed, bool verbose = false)
            {
                if(packed.size() % 2 == 0)
                    return true;

                if(verbose)
                    std::cout << "Range: expected an even number of packed bounds" << std::endl;
                return false;
            }
        };

        template <typename... Rules>
        struct RuleSet
        {
            using Tuple = std::tuple<Rules...>;

            constexpr RuleSet(Rules... rules)
                : rules(rules...)
            {
            }

            std::string description() const
            {
                return "RuleSet";
            }

            bool valid(bool verbose = false) const
            {
                return validImpl(verbose, std::index_sequence_for<Rules...>{});
            }

            template <typename ContextT>
            bool matches(const ContextT& context) const
            {
                return matchesImpl(context, std::index_sequence_for<Rules...>{});
            }

            Tuple rules;

        private:
            template <size_t... Indices>
            bool validImpl(bool verbose, std::index_sequence<Indices...>) const
            {
                return (... && std::get<Indices>(rules).valid(verbose));
            }

            template <typename ContextT, size_t... Indices>
            bool matchesImpl(const ContextT& context, std::index_sequence<Indices...>) const
            {
                return (... && std::get<Indices>(rules).matches(context));
            }
        };

        template <typename RuleSetT>
        struct Fallback
        {
            Fallback() = default;

            explicit Fallback(std::vector<RuleSetT> ruleSets)
                : rules(std::move(ruleSets))
            {
            }

            std::vector<RuleSetT> rules;

            std::string description() const
            {
                return "Fallback";
            }

            bool valid(bool verbose = false) const
            {
                for(const auto& rule : rules)
                {
                    if(!rule.valid(verbose))
                        return false;
                }
                return true;
            }

            template <typename ContextT>
            bool matches(const ContextT& context) const
            {
                for(const auto& rule : rules)
                {
                    if(rule.matches(context))
                        return true;
                }
                return false;
            }
        };
    } // namespace Fallback
} // namespace TensileLite

//Example usage:

/*

#include <iostream>
#include <vector>

#include "fallback.hpp"

namespace
{
	// Two tag templates let us define many unique keys while keeping only float/int families.
	template <int Id>
	struct FloatTag
	{
		using Type = float;
	};

	template <int Id>
	struct IntTag
	{
		using Type = int;
	};

	// Input dimensions/categories for this demo.
	using MTag     = FloatTag<0>;
	using NTag     = FloatTag<1>;
	using KTag     = FloatTag<2>;
	using ScoreTag = FloatTag<3>;
	using CatTag   = IntTag<0>;

	using CategoryRule = Fallback::Category<CatTag, int>;
	using MRule        = Fallback::Range<MTag, float>;
	using NRule        = Fallback::Range<NTag, float>;
	using KRule        = Fallback::Range<KTag, float>;
	using ScoreRule    = Fallback::Range<ScoreTag, float>;

	using PostRuleSet = Fallback::RuleSet<CategoryRule, MRule, NRule, KRule, ScoreRule>;
	using PostFallback = Fallback::Fallback<PostRuleSet>;

	// Build one rule-set: all rules in this set must match (AND semantics).
	PostRuleSet makeRuleSet(int ruleCategory,
							std::vector<float> mRanges,
							std::vector<float> nRanges,
							std::vector<float> kRanges,
							std::vector<float> scoreRanges)
	{
		return PostRuleSet(CategoryRule({ruleCategory}),
						   MRule(MRule::fromPairs(mRanges)),
						   NRule(NRule::fromPairs(nRanges)),
						   KRule(KRule::fromPairs(kRanges)),
						   ScoreRule(ScoreRule::fromPairs(scoreRanges)));
	}

	// Print a single evaluation to show usage clearly.
	void evaluate(const PostFallback& fb, float m, float n, float k, int cat, float score)
	{
		const auto context = Fallback::Context(Fallback::bind<MTag>(m),
											   Fallback::bind<NTag>(n),
											   Fallback::bind<KTag>(k),
											   Fallback::bind<CatTag>(cat),
											   Fallback::bind<ScoreTag>(score));

		const bool matched = fb.matches(context);

		std::cout << "INPUT: M=" << m << ", N=" << n << ", K=" << k << ", CAT=" << cat
				  << ", SCORE=" << score << " -> " << (matched ? "MATCH" : "NO MATCH")
				  << "\n";
	}
} // namespace

int main()
{
	// RuleSet #1: category=1, medium M/N/K, strong score.
	const auto rs1 = makeRuleSet(
		1,
		{128.0f, 1024.0f}, // M in (128, 1024)
		{128.0f, 1024.0f}, // N in (128, 1024)
		{64.0f, 4096.0f},  // K in (64, 4096)
		{0.80f, 1.01f});   // SCORE in (0.8, 1.01)

	// RuleSet #2: category=2, different range profile and score threshold.
	const auto rs2 = makeRuleSet(
		2,
		{32.0f, 256.0f},   // M in (32, 256)
		{32.0f, 256.0f},   // N in (32, 256)
		{16.0f, 1024.0f},  // K in (16, 1024)
		{0.60f, 0.95f});   // SCORE in (0.6, 0.95)

	// Fallback is OR across rule-sets: if rs1 OR rs2 matches, fallback triggers.
	const PostFallback fb({rs1, rs2});

	std::cout << "Fallback valid: " << (fb.valid(true) ? "true" : "false") << "\n\n";

	// Should match rs1.
	evaluate(fb, 512.0f, 512.0f, 2048.0f, 1, 0.92f);

	// Should match rs2.
	evaluate(fb, 128.0f, 64.0f, 512.0f, 2, 0.70f);

	// Should fail: category mismatch for both rule-sets.
	evaluate(fb, 512.0f, 512.0f, 2048.0f, 3, 0.92f);

	// Should fail: score too low for rs1 and too high-ranged dimensions for rs2.
	evaluate(fb, 512.0f, 512.0f, 2048.0f, 1, 0.40f);

	return 0;
}

*/