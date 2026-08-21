from manim import *


class PrefillVsDecode(Scene):
    """Teach the difference between prompt prefill and token-by-token decode."""

    def token_box(self, label: str, color=BLUE_D):
        box = RoundedRectangle(
            corner_radius=0.08,
            width=1.25,
            height=0.62,
            stroke_color=color,
            fill_color=color,
            fill_opacity=0.25,
        )
        text = Text(label, font_size=25)
        return VGroup(box, text)

    def construct(self):
        title = Text("LLM Generation: Prefill vs Decode", font_size=38)
        title.to_edge(UP)
        self.play(Write(title))

        prompt_label = Text("Prompt prefill: process the whole prompt", font_size=25)
        prompt_label.move_to(UP * 2.05 + LEFT * 2.8)

        prompt = VGroup(
            self.token_box("The"),
            self.token_box("cat"),
            self.token_box("sat"),
        ).arrange(RIGHT, buff=0.18)
        prompt.next_to(prompt_label, DOWN, buff=0.22)

        self.play(FadeIn(prompt_label), LaggedStart(*[FadeIn(t) for t in prompt]))

        arrow = Arrow(
            prompt.get_right(),
            RIGHT * 3.9 + UP * 0.9,
            buff=0.2,
            stroke_width=5,
            color=YELLOW,
        )
        attention = Text("Transformer layers", font_size=25, color=YELLOW)
        attention.next_to(arrow, UP, buff=0.12)
        self.play(GrowArrow(arrow), FadeIn(attention))

        cache_label = Text("KV cache", font_size=25, color=GREEN)
        cache_label.move_to(DOWN * 0.05 + LEFT * 3.25)
        cache = VGroup(
            self.token_box("K/V 0", GREEN_D),
            self.token_box("K/V 1", GREEN_D),
            self.token_box("K/V 2", GREEN_D),
        ).arrange(RIGHT, buff=0.12)
        cache.next_to(cache_label, DOWN, buff=0.2)
        self.play(FadeIn(cache_label), LaggedStart(*[FadeIn(t) for t in cache]))

        prefill_note = Text(
            "All prompt tokens create attention history",
            font_size=22,
            color=GREEN_B,
        )
        prefill_note.next_to(cache, DOWN, buff=0.25)
        self.play(Write(prefill_note))
        self.wait(1)

        divider = Line(LEFT * 6.2, RIGHT * 6.2, color=GRAY).shift(DOWN * 1.65)
        self.play(Create(divider))

        decode_label = Text("Decode: process one new token at a time", font_size=25)
        decode_label.move_to(UP * 1.15 + LEFT * 2.35)
        self.play(FadeIn(decode_label))

        next_token = self.token_box("on", ORANGE)
        next_token.move_to(RIGHT * 3.75 + DOWN * 2.35)
        decode_arrow = Arrow(
            next_token.get_left(),
            cache.get_right() + DOWN * 2.45,
            buff=0.2,
            stroke_width=5,
            color=ORANGE,
        )
        self.play(FadeIn(next_token), GrowArrow(decode_arrow))

        new_cache_item = self.token_box("K/V 3", GREEN_D)
        new_cache_item.next_to(cache, RIGHT, buff=0.12)
        self.play(
            next_token.animate.move_to(new_cache_item.get_center()),
            FadeOut(decode_arrow),
        )
        self.play(Transform(next_token, new_cache_item))

        generated = Text("Next token: 'the'", font_size=25, color=ORANGE)
        generated.move_to(RIGHT * 3.7 + DOWN * 1.75)
        self.play(Write(generated))

        for label, index in [("the", 4), ("mat", 5)]:
            token = self.token_box(label, ORANGE)
            token.move_to(RIGHT * 3.75 + DOWN * 2.35)
            self.play(FadeIn(token))
            cache_item = self.token_box(f"K/V {index}", GREEN_D)
            cache_item.next_to(cache, RIGHT, buff=0.12)
            self.play(token.animate.move_to(cache_item.get_center()))
            self.play(Transform(token, cache_item))

        final_note = Text(
            "Prefill is wide; decode is incremental and cache-heavy.",
            font_size=26,
            color=YELLOW,
        )
        final_note.to_edge(DOWN)
        self.play(Write(final_note))
        self.wait(2)

