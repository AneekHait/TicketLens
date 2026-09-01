"""Presentation layer for the TicketLens GUI.

`gui.py` is the controller: it owns the data, the worker threads and every
signal connection. The widget construction lives here, split by kind:

* `theme`   -- palette, global stylesheet, widget factories (no dependencies)
* `widgets` -- sidebar, activity-log console, numeric-sort tree item
* `dialogs` -- the modal dialogs
* `pages`   -- the four main pages

The dependency direction is one-way: theme <- widgets <- dialogs, theme <- pages,
and gui.py imports from all of them. Nothing here imports gui.py, so there is no
cycle to work around.
"""
