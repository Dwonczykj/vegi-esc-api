from vegi_esc_api.dash_apps import create_dash_app

# endpoint of this page
URL_RULE = "/custom-app"
# dash internal route prefix, must be start and end with "/"
URL_BASE_PATHNAME = "/dash/custom-app/"


def create_dash(server):
    """Create a Dash view"""
    app = create_dash_app(server, URL_RULE, URL_BASE_PATHNAME)

    # dash app definitions goes here

    return app.server
