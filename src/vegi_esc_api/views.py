from flask import Blueprint, render_template

base_app = Blueprint("base_app", __name__)

top_menus = [
    {"path ": "/", "title": "Home"},
    {"path ": "/experiments", "title" : "Experiments"},
    {"path ": "/custom-app " , "title" : "Custom App"},
    # ...
]


def get_top_menu_items(route: str):
    return top_menus


@base_app.route("/")
def index():
    """Landing page."""
    return render_template("index.html", top_menu_items=get_top_menu_items("/"))
