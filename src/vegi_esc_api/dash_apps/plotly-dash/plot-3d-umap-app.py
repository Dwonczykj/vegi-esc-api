# Import packages
from dash import Dash, html, dash_table, dcc
# from umap import UMAP
from sklearn.manifold import TSNE
import plotly.express as px
import pandas as pd


df_iris = px.data.iris()

features = df_iris.loc[:, :'petal_width']

tsne = TSNE(n_components=2, random_state=0)
projections = tsne.fit_transform(features)

fig_2d = px.scatter(
    projections, x=0, y=1,
    color=df_iris.species, labels={'color': 'species'}
)

tsne = TSNE(n_components=3, random_state=0)
projections = tsne.fit_transform(features, )

fig_3d = px.scatter_3d(
    projections, x=0, y=1, z=2,
    color=df_iris.species, labels={'color': 'species'}
)
fig_3d.update_traces(marker_size=8)


# Initialize the app
app = Dash(__name__)


# features = df.loc[:, :'petal_width']

# umap_2d = UMAP(n_components=2, init='random', random_state=0)
# umap_3d = UMAP(n_components=3, init='random', random_state=0)

# proj_2d = umap_2d.fit_transform(features)
# proj_3d = umap_3d.fit_transform(features)

# fig_2d = px.scatter(
#     proj_2d, x=0, y=1,
#     color=df_iris.species, labels={'color': 'species'}
# )
# fig_3d = px.scatter_3d(
#     proj_3d, x=0, y=1, z=2,
#     color=df_iris.species, labels={'color': 'species'}
# )
# fig_3d.update_traces(marker_size=5)

# # For use in scripts and notebooks:
# # fig_2d.show()
# # fig_3d.show()

# Incorporate data
df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminder2007.csv')

# App layout
app.layout = html.Div([
    html.Div(children='My First App with Data and a Graph'),
    html.H1(children='3D Visualisation'),
    dcc.Graph(
        figure=fig_3d
    ),
    html.H1(children='2D Visualisation'),
    dcc.Graph(
        figure=fig_2d
    ),
    html.H1(children='GapMinder2007 Dataset Example'),
    dash_table.DataTable(data=df.to_dict('records'), page_size=10),
    dcc.Graph(
        figure=px.histogram(df, x='continent', y='lifeExp', histfunc='avg')
    ),
])


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)

    # ```shell
    # python src/vegi_esc_api/dash_apps/plotly-dash/plot-3d-umap-app.py
    # ```
