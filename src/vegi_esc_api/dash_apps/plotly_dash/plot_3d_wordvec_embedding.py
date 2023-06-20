# Import packages
from flask import Flask
from vegi_esc_api.sustained import SustainedAPI
from vegi_esc_api.word_vec_model import getModel
import vegi_esc_api.logger as logger
import cachetools.func
import re
import pickle
from dash import html, dash_table, dcc, Output, Input, Dash, State, callback

# from umap import UMAP
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objs as go
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd


# # endpoint of this page
# URL_RULE = "/custom-app"
# # dash internal route prefix, must be start and end with "/"
# URL_BASE_PATHNAME = "/dash/custom-app/"

model = getModel()


# @cachetools.func.ttl_cache(maxsize=128, ttl=10 * 60)
def append_list(sim_words, words):
    list_of_words = []

    for i in range(len(sim_words)):
        sim_words_list = list(sim_words[i])
        sim_words_list.append(words)
        sim_words_tuple = tuple(sim_words_list)
        list_of_words.append(sim_words_tuple)

    return list_of_words


def get_word_vectors(
    words=None,
    sample=10,
):
    if words is None:
        if sample > 0:
            words = np.random.choice(list(model.key_to_index.keys()), sample)
        else:
            words = [word for word in model.vocab]

    vocab = model.key_to_index.keys()
    if all((phrase in vocab for phrase in words)):
        word_vectors = np.array([model[w] for w in words])
        return word_vectors
    else:
        with open('tfidf_vectorizer.pk', 'wb') as fin:
            tfidf_vectorizer = pickle.load(fin)
        tfidf_weighted_phrase_vects = [
            (
                (1 / tfidf_vectorizer.transform([phrase])[0].T.todense())
                * np.sum([w for w in re.split(r'\s', phrase)])
            )
            for phrase in words]
        phrase_vectors = tfidf_weighted_phrase_vects
        return phrase_vectors


def display_scatterplot_3D(
    model,
    user_input=None,
    words=None,
    label=None,
    color_map=None,
    annotation="On",
    dim_red="PCA",
    perplexity=0,
    learning_rate=0,
    iteration=0,
    topn=0,
    sample=10,
):
    if user_input is None:
        user_input = ""
    word_vectors = get_word_vectors(words=words, sample=sample)

    if dim_red == "PCA":
        three_dim = PCA(random_state=0).fit_transform(word_vectors)[:, :3]
    else:
        three_dim = TSNE(
            n_components=3,
            random_state=0,
            perplexity=perplexity,
            learning_rate=learning_rate,
            n_iter=iteration,
        ).fit_transform(word_vectors)[:, :3]

    color = "blue"
    quiver = go.Cone(
        x=[0, 0, 0],
        y=[0, 0, 0],
        z=[0, 0, 0],
        u=[1.5, 0, 0],
        v=[0, 1.5, 0],
        w=[0, 0, 1.5],
        anchor="tail",
        colorscale=[[0, color], [1, color]],
        showscale=False,
    )

    data = [quiver]

    count = 0
    for i in range(len(user_input)):
        trace = go.Scatter3d(
            x=three_dim[count : count + topn, 0],
            y=three_dim[count : count + topn, 1],
            z=three_dim[count : count + topn, 2],
            text=words[count : count + topn] if annotation == "On" else "",
            name=user_input[i],
            textposition="top center",
            textfont_size=16,
            mode="markers+text",
            marker={"size": 10, "opacity": 0.8, "color": 2},
        )

        data.append(trace)
        count = count + topn

    trace_input = go.Scatter3d(
        x=three_dim[count:, 0],
        y=three_dim[count:, 1],
        z=three_dim[count:, 2],
        text=words[count:],
        name="input words",
        textposition="top center",
        textfont_size=12,
        mode="markers+text",
        marker={"size": 10, "opacity": 1, "color": "black"},
    )

    data.append(trace_input)

    # Configure the layout.
    layout = go.Layout(
        margin={"l": 0, "r": 0, "b": 0, "t": 0},
        showlegend=True,
        legend=dict(
            x=1, y=0.5, font=dict(family="Courier New", size=25, color="black")
        ),
        font=dict(family=" Courier New ", size=14),
        autosize=False,
        width=1000,
        height=1000,
    )

    plot_figure = go.Figure(data=data, layout=layout)
    return plot_figure
    # st.plotly_chart(plot_figure)
    # return dcc.Graph(figure=plot_figure)


# def horizontal_bar(word, similarity):
#     similarity = [round(elem, 2) for elem in similarity]

#     data = go.Bar(
#         x=similarity,
#         y=word,
#         orientation="h",
#         text=similarity,
#         marker_color=4,
#         textposition="auto",
#     )

#     layout = go.Layout(
#         font=dict(size=20),
#         xaxis=dict(showticklabels=False, automargin=True),
#         yaxis=dict(showticklabels=True, automargin=True, autorange="reversed"),
#         margin=dict(t=20, b=20, r=10),
#     )

#     plot_figure = go.Figure(data=data, layout=layout)
#     st.plotly_chart(plot_figure)


def display_scatterplot_2D(
    model,
    user_input=None,
    words=None,
    label=None,
    color_map=None,
    annotation="On",
    dim_red="PCA",
    perplexity=0,
    learning_rate=0,
    iteration=0,
    topn=0,
    sample=10,
):
    if user_input is None:
        user_input = ""

    word_vectors = get_word_vectors(words=words, sample=sample)

    if dim_red == "PCA":
        two_dim = PCA(random_state=0).fit_transform(word_vectors)[:, :2]
    else:
        two_dim = TSNE(
            random_state=0,
            perplexity=perplexity,
            learning_rate=learning_rate,
            n_iter=iteration,
        ).fit_transform(word_vectors)[:, :2]

    data = []
    count = 0
    for i in range(len(user_input)):
        trace = go.Scatter(
            x=two_dim[count : count + topn, 0],
            y=two_dim[count : count + topn, 1],
            text=words[count : count + topn] if annotation == "On" else "",
            name=user_input[i],
            textposition="top center",
            textfont_size=16,
            mode="markers+text",
            marker={"size": 15, "opacity": 0.8, "color": 2},
        )

        data.append(trace)
        count = count + topn

    trace_input = go.Scatter(
        x=two_dim[count:, 0],
        y=two_dim[count:, 1],
        text=words[count:],
        name="input words",
        textposition="top center",
        textfont_size=12,
        mode="markers+text",
        marker={"size": 25, "opacity": 1, "color": "black"},
    )

    data.append(trace_input)

    # Configure the layout.
    layout = go.Layout(
        margin={"l": 0, "r": 0, "b": 0, "t": 0},
        showlegend=True,
        hoverlabel=dict(bgcolor="white", font_size=20, font_family="Courier New"),
        legend=dict(
            x=1, y=0.5, font=dict(family="Courier New", size=25, color="black")
        ),
        font=dict(family=" Courier New ", size=14),
        autosize=False,
        width=1000,
        height=1000,
    )

    plot_figure = go.Figure(data=data, layout=layout)
    return plot_figure
    # st.plotly_chart(plot_figure)


# df_iris = px.data.iris()

# features = df_iris.loc[:, :"petal_width"]

# tsne = TSNE(n_components=2, random_state=0)
# projections = tsne.fit_transform(features)

# fig_2d = px.scatter(
#     projections, x=0, y=1, color=df_iris.species, labels={"color": "species"}
# )

# tsne = TSNE(n_components=3, random_state=0)
# projections = tsne.fit_transform(
#     features,
# )

# fig_3d = px.scatter_3d(
#     projections, x=0, y=1, z=2, color=df_iris.species, labels={"color": "species"}
# )
# fig_3d.update_traces(marker_size=8)

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


def update_figure(
    dim_red="TSNE", dimension="2D", top_n=10, annotation="On", user_input="", view_to_similar_words: list[str] = [],
):
    # dim_red = st.sidebar.selectbox("Select dimension reduction method", ("PCA", "TSNE"))

    # dimension = st.sidebar.radio("Select the dimension of the visualization", ("2D", "3D"))

    # user_input = st.sidebar.text_input(
    #     "Type the word that you want to investigate. You can type more than one word by separating one word with other with comma (,)",
    #     "",
    # )

    # top_n = st.sidebar.slider(
    #     "Select the amount of words associated with the input words you want to visualize ",
    #     5,
    #     100,
    #     (5),
    # )

    # annotation = st.sidebar.radio(
    #     "Enable or disable the annotation on the visualization", ("On", "Off")
    # )

    if dim_red == "TSNE":
        perplexity = max(min(5, top_n), 0)
        # perplexity = st.sidebar.slider(
        #     "Adjust the perplexity. The perplexity is related to the number of nearest neighbors that is used in other manifold learning algorithms. Larger datasets usually require a larger perplexity",
        #     5,
        #     50,
        #     (30),
        # )

        learning_rate = 200
        # learning_rate = st.sidebar.slider("Adjust the learning rate", 10, 1000, (200))

        iteration = 1000
        # iteration = st.sidebar.slider("Adjust the number of iteration", 250, 100000, (1000))

    else:
        perplexity = 0
        learning_rate = 0
        iteration = 0

    if user_input == "":
        similar_word = None
        labels = None
        color_map = None

    else:
        user_input = [x.strip() for x in user_input.split(",")]
        result_word = []

        for words in user_input:
            if not view_to_similar_words:
                sim_words = model.most_similar(words, topn=top_n)
                sim_words = append_list(sim_words, words)
            else:
                sim_words = []
                vocab = model.key_to_index.keys()
                if all((phrase in vocab for phrase in view_to_similar_words)):
                    for measure_similarity_to_word in view_to_similar_words:
                        _similarity = model.similarity(words, measure_similarity_to_word)
                        sim_words.append((measure_similarity_to_word, _similarity, words))
                    sim_words.sort(key=lambda t: t[1])
                else:
                    for measure_similarity_to_word in view_to_similar_words:
                        _similarity = model.wmdistance(words, measure_similarity_to_word)
                        # sim_words.append((measure_similarity_to_word, _similarity, words))
                        sim_words.append((measure_similarity_to_word, 1.0 / _similarity, words))
                    # sim_words.sort(key=lambda t: t[1], reverse=True)
                    sim_words.sort(key=lambda t: t[1])
                sim_words = sim_words[:top_n]
            result_word.extend(sim_words)

        similar_word = [word[0] for word in result_word]
        # similarity = [word[1] for word in result_word]
        similar_word.extend(user_input)
        labels = [word[2] for word in result_word]
        label_dict = dict([(y, x + 1) for x, y in enumerate(set(labels))])
        color_map = [label_dict[x] for x in labels]
    # Initialize the app
    if dimension == "2D":
        plot = display_scatterplot_2D(
            model,
            user_input,
            similar_word,
            labels,
            color_map,
            annotation,
            dim_red,
            perplexity,
            learning_rate,
            iteration,
            top_n,
        )
    else:
        plot = display_scatterplot_3D(
            model,
            user_input,
            similar_word,
            labels,
            color_map,
            annotation,
            dim_red,
            perplexity,
            learning_rate,
            iteration,
            top_n,
        )
    return plot


# Incorporate data
df = pd.read_csv(
    "https://raw.githubusercontent.com/plotly/datasets/master/gapminder2007.csv"
)

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

# categories = [
#     'pasta',
#     'tomato',
#     'rice',
#     'pizza',
#     'chocolate',
#     'bread',
#     'toffee',
#     'coke',
#     'beans',
#     'honey',
#     'yoghurt',
#     'milk',
#     'whey',
#     'broccoli',
#     'vegetable'
# ]


def get_data(app: Flask):
    ss = SustainedAPI(app=app)
    categories = ss.get_categories()
    categories = [c.name for c in categories]
    return {
        'categories': categories
    }


layout = html.Div([
    dcc.Input(id="input-1", type="text", value="Montréal"),
    dcc.Input(id="input-2", type="text", value="Canada"),
    html.Div(id="number-output"),
    html.Div(className='row', children=[
            html.Div(className='six columns', children=[
                dash_table.DataTable(data=df.to_dict('records'), page_size=11, style_table={'overflowX': 'auto'})
            ]),
            html.Div(className='six columns', children=[
                dcc.Graph(figure={}, id='histo-chart-final')
            ])
    ]),
    html.Hr(),
    html.H1(children="My First App with Data and a Graph"),
    html.Div(
        children="Type the word that you want to investigate. You can type more than one word by separating one word with other with comma (,)"
    ),
    html.Label("User Input"),
    dcc.Input(id="input-3", type="text", value="Pasta"),
    html.Button(id="update-user-input-button", n_clicks=0, children="Run"),
    html.Br(),
    html.Label("Dimensions"),
    dcc.RadioItems(["2D", "3D"], "3D", id="radio-dimensions"),
    html.Label("Use Food Categories"),
    dcc.RadioItems(["Any", "Categories"], "Any", id="word-dict"),
    html.Label("Dimensionality Reduction Technique"),
    dcc.RadioItems(["TSNE", "PCA"], "TSNE", id="radio-dim-reduction"),
    html.Br(),
    html.Label("Number of Similar words"),
    dcc.Slider(
        id="num-similar-words",
        min=0,
        max=100,
        marks={i: f"{i} words" for i in [j * 5 for j in range(0, 21)]},
        value=10,
    ),
    html.Br(),
    html.H2(id="fig-header"),
    html.Div(className='row', children=[
            # html.Div(className='six columns', children=[
            #     dash_table.DataTable(data=df.to_dict('records'), page_size=11, style_table={'overflowX': 'auto'})
            # ]),
            html.Div(children=[
                dcc.Graph(figure={}, id='word-vec-fig')
            ])
    ]),
])


def register_callbacks(server: Flask, dashapp: Dash):
    data = get_data(app=server)
    categories = data['categories']

    @dashapp.callback(
        Output("histo-chart-final", "figure"),
        Input("input-1", "value"),
        Input("input-2", "value"),
    )
    def update_figure_output(input1, input2):
        print(u'Input 1 is "{}" and Input 2 is "{}"'.format(input1, input2))
        if len(input2) > len(input1):
            fig = px.histogram(df, x='continent', y='lifeExp', histfunc='avg')
            return fig
        else:
            fig = px.histogram(df, x='continent', y='gdpPercap', histfunc='avg')
            return fig

    @dashapp.callback(
        Output("word-vec-fig", "figure"),
        Input("input-3", "value"),
        Input("radio-dimensions", "value"),
        Input("num-similar-words", "value"),
        Input("word-dict", "value"),
        Input("radio-dim-reduction", "value"),
    )
    def update_word_viz_output(input3, radioValue, top_n, word_type, dim_reduction):
        print(u'Input 3 is "{}" and radio input value is "{}"'.format(input3, radioValue))
        fig = update_figure(dimension=radioValue, dim_red=dim_reduction, user_input=input3, top_n=top_n, view_to_similar_words=(categories if word_type != 'Any' else []))
        return fig


# @callback(
#     Output("number-output", "children"),
#     Input("input-1", "value"),
#     Input("input-2", "value"),
# )
# def update_output(input1, input2):
#     return u'Input 1 is "{}" and Input 2 is "{}"'.format(input1, input2)




# plot = {}



# layout = html.Div(
#     [
#         html.H1(children="My First App with Data and a Graph"),
#         html.Div(
#             children="Type the word that you want to investigate. You can type more than one word by separating one word with other with comma (,)"
#         ),
#         html.Label("User Input"),
#         dcc.Input(id="user-input", value="Canellini beans", type="text"),
#         html.Button(id="update-user-input-button", n_clicks=0, children="Run"),
#         html.Br(),
#         html.Label("Dimensions"),
#         dcc.RadioItems(["2D", "3D"], "3D", id="radio-dimensions"),
#         html.Br(),
#         html.Label("Number of Similar words"),
#         dcc.Slider(
#             id="num-similar-words",
#             min=0,
#             max=100,
#             marks={i: f"Label {i}" if i == 1 else str(i) for i in range(1, 6)},
#             value=10,
#         ),
#         html.Br(),
#         html.H2(id="fig-header"),
#         dcc.Graph(id="word-vec-fig"),
#         html.Div(id="fig-desc"),
#         # html.H3(children="3D Visualisation"),
#         # dcc.Graph(figure=fig_3d),
#         # html.H1(children="2D Visualisation"),
#         # dcc.Graph(figure=fig_2d),
#         html.H1(children="GapMinder2007 Dataset Example"),
#         html.Hr(),
#         # dcc.RadioItems(options=['pop', 'lifeExp', 'gdpPercap'], value='lifeExp', id='controls-and-radio-item'),
#         # dash_table.DataTable(data=df.to_dict("records"), page_size=10),
#         # dcc.Graph(figure=px.histogram(df, x="continent", y="lifeExp", histfunc="avg")),
#         html.Div(className='row', children='My First App with Data, Graph, and Controls',
#                  style={'textAlign': 'center', 'color': 'blue', 'fontSize': 30}),

#         html.Div(className='row', children=[
#             dcc.RadioItems(options=['pop', 'lifeExp', 'gdpPercap'],
#                            value='lifeExp',
#                            inline=True,
#                            id='my-radio-buttons-final')
#         ]),
#         html.Div(className='row', children=[
#             html.Div(className='six columns', children=[
#                 dash_table.DataTable(data=df.to_dict('records'), page_size=11, style_table={'overflowX': 'auto'})
#             ]),
#             html.Div(className='six columns', children=[
#                 dcc.Graph(figure={}, id='histo-chart-final')
#             ])
#         ])
#     ]
# )



# @callback(
#     Output(component_id='histo-chart-final', component_property='figure'),
#     Input(component_id='my-radio-buttons-final', component_property='value')
# )
# def update_graph_for_radio(col_chosen: str):
#     fig = px.histogram(df, x='continent', y=col_chosen, histfunc='avg')
#     print('HELLO')
#     return fig


# def register_callbacks(dashapp: Dash):
#     # Add controls to build the interaction
#     # @dashapp.callback(
#     #     Output(component_id='histo-chart-final', component_property='figure'),
#     #     Input(component_id='my-radio-buttons-final', component_property='value')
#     # )
#     # def update_graph_for_radio(col_chosen: str):
#     #     fig = px.histogram(df, x='continent', y=col_chosen, histfunc='avg')
#     #     return fig

#     @dashapp.callback(
#         Output(component_id="fig-header", component_property="children"),
#         Input(component_id="radio-dimensions", component_property="value"),
#     )
#     def update_fig_header(radio_dims_value: str):
#         return f"{radio_dims_value} Visualisation"

#     @dashapp.callback(
#         Output(component_id="word-vec-fig", component_property="figure"),
#         Input(component_id="radio-dimensions", component_property="value"),
#     )
#     def update_fig_dimensions(radio_dims_value: str):
#         return update_figure(dimension=radio_dims_value)

#     @dashapp.callback(
#         Output(component_id="word-vec-fig", component_property="figure"),
#         Input(component_id="num-similar-words", component_property="value"),
#     )
#     def update_fig_top_n(top_n_value: int):
#         return update_figure(top_n=top_n_value)

#     @dashapp.callback(
#         Output(component_id="fig-desc", component_property="children"),
#         Input(component_id="user-input", component_property="value"),
#     )
#     def update_output_div(user_input_value):
#         # return f'Output: {user_input_value}'
#         logger.info(f"Updated user input value{user_input_value}")
#         return f'Visualisation for words around "{user_input_value}"'

#     # @dashapp.callback(
#     #     Output(component_id="word-vec-fig", component_property="figure"),
#     #     Input(component_id="user-input", component_property="value"),
#     # )
#     # def update_output_div(user_input_value):
#     #     # return f'Output: {user_input_value}'
#     #     logger.info(f'Updated user input value{user_input_value}')
#     #     return update_figure(user_input=user_input_value)

#     # * multiple callbacks for one input change
#     # @dashapp.callback(
#     #     Output('square', 'children'),
#     #     Output('cube', 'children'),
#     #     Output('twos', 'children'),
#     #     Output('threes', 'children'),
#     #     Output('x^x', 'children'),
#     #     Input('num-multi', 'value'))
#     # def callback_a(x):
#     #     return (x**2, x**3, 2**x, 3**x, x**x)

#     # @dashapp.callback(
#     #     Output('output-state', 'children'),
#     #     Input('submit-button-state', 'n_clicks'),
#     #     State('input-1-state', 'value'),
#     #     State('input-2-state', 'value'))
#     # def update_output(n_clicks, input1, input2):
#     #     return u'''
#     #         The Button has been pressed {} times,
#     #         Input 1 is "{}",
#     #         and Input 2 is "{}"
#     #     '''.format(n_clicks, input1, input2)
#     @dashapp.callback(
#         Output(component_id="word-vec-fig", component_property="figure"),
#         Input("update-user-input-button", "n_clicks"),
#         # State('input-1-state', 'value'),
#         State("user-input", "value"),
#     )
#     def update_output(n_clicks: int, user_input_value: str):
#         logger.log(
#             """
#             The Button has been pressed {} times,
#             Input 1 is "{}"
#         """.format(
#                 n_clicks, user_input_value
#             )
#         )
#         return update_figure(user_input=user_input_value)
