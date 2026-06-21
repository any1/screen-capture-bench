#!/usr/bin/env python3
"""Compile an HTML report from screen-capture-bench JSON output.

The report shows fps as a table per GPU, with buffer modifiers down the
vertical axis and the remaining properties (mapping, shadow buffer) across
the horizontal axis. Rows are ordered by performance, best fps at the top.

Usage:
    report.py [-o report.html] < results.json

The JSON is read from standard input. With no output file the HTML is
written to stdout.
"""

import argparse
import html
import json
import sys


def column_label(result):
	"""Build the horizontal-axis label for a result's properties.

	GBM mapping never uses a shadow buffer, so the shadow state is implied
	and left off; only Vulkan mapping spells it out.
	"""
	mapping = result.get("mapping", "?")
	if mapping == "gbm":
		return mapping
	shadow = "shadow" if result.get("shadow_buffer") else "no shadow"
	return "%s / %s" % (mapping, shadow)


def row_label(result):
	"""Build the vertical-axis label for a result's modifier."""
	label = result.get("modifier", "?")
	if result.get("optimal_tiling"):
		label += " *"
	return label


def collect_axes(results):
	"""Return ordered (rows, columns) keyed by modifier and property combo.

	Columns keep first-seen order. Rows are ordered by the best fps achieved
	for that modifier so the fastest modifier sits at the top.
	"""
	columns = []
	rows = {}

	for result in results:
		col = column_label(result)
		if col not in columns:
			columns.append(col)

		row = row_label(result)
		fps = result.get("fps")
		best = rows.get(row, -1.0)
		if fps is not None and fps > best:
			rows[row] = fps
		elif row not in rows:
			rows[row] = best

	ordered_rows = sorted(rows, key=lambda r: rows[r], reverse=True)
	return ordered_rows, columns


def build_cell_index(results):
	"""Map (row label, column label) -> result for quick lookup.

	Several raw results can collapse onto one cell (e.g. GBM with and
	without a shadow buffer both label as "gbm"). Keep the most useful one:
	a real measurement always wins over a skipped combination.
	"""
	index = {}
	for result in results:
		key = (row_label(result), column_label(result))
		existing = index.get(key)
		if existing is None or (existing.get("skipped") and not result.get("skipped")):
			index[key] = result
	return index


def cell_html(result):
	"""Render a single table cell from a result, or empty for a gap."""
	if result is None:
		return "<td></td>"
	if result.get("skipped"):
		reason = html.escape(result.get("skip_reason", "skipped"))
		return '<td title="%s"></td>' % reason
	fps = result.get("fps")
	if fps is None:
		return "<td></td>"
	return "<td>%.1f</td>" % fps


def gpu_table_html(gpu):
	name = html.escape(gpu.get("gpu", "Unknown GPU"))
	results = gpu.get("results", [])

	rows, columns = collect_axes(results)
	index = build_cell_index(results)

	parts = ["<h2>%s</h2>" % name]
	parts.append("<table border=\"1\" cellpadding=\"6\" cellspacing=\"0\">")

	header = ["<th>Modifier</th>"]
	for col in columns:
		header.append("<th>%s</th>" % html.escape(col))
	parts.append("<tr>" + "".join(header) + "</tr>")

	for row in rows:
		cells = ["<th>%s</th>" % html.escape(row)]
		for col in columns:
			cells.append(cell_html(index.get((row, col))))
		parts.append("<tr>" + "".join(cells) + "</tr>")

	parts.append("</table>")
	return "\n".join(parts)


def system_html(system):
	"""Render the generic system information as a small table."""
	if not system:
		return ""

	mem = system.get("memory_bytes")
	mem_str = "%.1f GiB" % (mem / 1024**3) if mem else None
	fields = [
		("Distro", system.get("distro")),
		("Kernel", system.get("kernel")),
		("CPU", system.get("cpu")),
		("Threads", system.get("cpu_threads")),
		("Memory", mem_str),
	]

	parts = ["<h2>System</h2>",
		 "<table border=\"1\" cellpadding=\"6\" cellspacing=\"0\">"]
	for label, value in fields:
		if value in (None, ""):
			continue
		parts.append("<tr><th>%s</th><td>%s</td></tr>"
			     % (label, html.escape(str(value))))
	parts.append("</table>")
	return "\n".join(parts)


def build_report(data):
	# The root is an object with system info and a list of GPUs; older
	# output was a bare list of GPUs.
	if isinstance(data, dict):
		system = data.get("system", {})
		gpus = data.get("gpus", [])
	else:
		system = {}
		gpus = data

	parts = [
		"<!DOCTYPE html>",
		"<html>",
		"<head><title>Screen capture benchmark report</title></head>",
		"<body>",
		"<h1>Screen capture benchmark report</h1>",
		"<p>Values are frames per second. "
		"A <code>*</code> marks the Vulkan-optimal modifier. "
		"Empty cells are skipped combinations (hover for the reason).</p>",
	]

	system_section = system_html(system)
	if system_section:
		parts.append(system_section)

	for gpu in gpus:
		parts.append(gpu_table_html(gpu))

	parts.append("</body>")
	parts.append("</html>")
	return "\n".join(parts)


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("-o", "--output", help="HTML output file (default: stdout)")
	args = parser.parse_args()

	data = json.load(sys.stdin)

	report = build_report(data)

	if args.output:
		with open(args.output, "w") as f:
			f.write(report + "\n")
	else:
		sys.stdout.write(report + "\n")


if __name__ == "__main__":
	main()
