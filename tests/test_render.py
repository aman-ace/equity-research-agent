from equity_agent import render


class TestBlocks:
    def test_headings_keep_their_level(self):
        html = render.to_html("# Title\n\n## Section\n\n### Detail")
        assert "<h1>Title</h1>" in html
        assert "<h2>Section</h2>" in html
        assert "<h3>Detail</h3>" in html

    def test_horizontal_rule(self):
        assert "<hr>" in render.to_html("a\n\n---\n\nb")

    def test_wrapped_lines_join_into_one_paragraph(self):
        html = render.to_html("Costco operates 914\nwarehouses worldwide.")
        assert html == "<p>Costco operates 914 warehouses worldwide.</p>"

    def test_a_heading_ends_a_paragraph(self):
        html = render.to_html("Intro text.\n## Next")
        assert html == "<p>Intro text.</p>\n<h2>Next</h2>"

    def test_numbered_list_becomes_an_ordered_list(self):
        html = render.to_html("1. first\n2. second")
        assert html == "<ol><li>first</li><li>second</li></ol>"

    def test_bulleted_list_becomes_an_unordered_list(self):
        html = render.to_html("- one\n- two")
        assert html == "<ul><li>one</li><li>two</li></ul>"

    def test_table_is_rendered_with_a_header(self):
        html = render.to_html("| Year | Revenue |\n|---|---|\n| 2025 | $275.2bn |")
        assert "<th>Year</th><th>Revenue</th>" in html
        assert "<td>2025</td><td>$275.2bn</td>" in html

    def test_a_pipe_row_without_a_separator_is_not_a_table(self):
        # Prose that happens to contain a pipe must not start a table -- and must
        # still be consumed, or the block loop spins forever.
        html = render.to_html("| not really a table")
        assert "<table>" not in html
        assert html == "<p>| not really a table</p>"

    def test_a_hash_without_a_space_is_not_a_heading(self):
        assert render.to_html("#notaheading") == "<p>#notaheading</p>"

    def test_every_line_is_consumed(self):
        """Regression: a block the dispatcher declines must still advance.

        A pipe-prefixed line also ends the paragraph before it, so these are
        three separate paragraphs rather than one run-together block.
        """
        html = render.to_html("| a\n#b\ntext\n\n| c")
        assert html == "<p>| a</p>\n<p>#b text</p>\n<p>| c</p>"

    def test_blocks_appear_in_source_order(self):
        html = render.to_html("## A\n\ntext\n\n| x |\n|---|\n| 1 |")
        assert html.index("<h2>") < html.index("<p>") < html.index("<table>")


class TestInline:
    def test_bold_and_italic(self):
        assert (
            render.inline("**bold** and *slanted*") == "<strong>bold</strong> and <em>slanted</em>"
        )

    def test_code_span(self):
        assert render.inline("run `pytest` now") == "run <code>pytest</code> now"

    def test_autolinked_url(self):
        rendered = render.inline("<https://sec.gov/x>")
        assert 'href="https://sec.gov/x"' in rendered
        assert 'rel="noopener noreferrer"' in rendered

    def test_labelled_link_keeps_its_label(self):
        rendered = render.inline("[EDGAR](https://sec.gov)")
        assert ">EDGAR</a>" in rendered
        assert 'href="https://sec.gov"' in rendered


class TestEscaping:
    def test_markup_in_the_source_is_escaped(self):
        # A filing quote containing a tag must not become live markup.
        assert render.inline("<script>alert(1)</script>") == "&lt;script&gt;alert(1)&lt;/script&gt;"

    def test_ampersands_are_escaped(self):
        assert render.inline("Procter & Gamble") == "Procter &amp; Gamble"

    def test_escaping_survives_a_table_cell(self):
        html = render.to_html("| a |\n|---|\n| <b>x</b> |")
        assert "&lt;b&gt;x&lt;/b&gt;" in html
        assert "<b>" not in html
