from vegi_esc_api.create_app import create_app
from vegi_esc_api.sustained import SustainedAPI

if __name__ == "__main__":
    app, vegi_db_session = create_app(None)
    ss = SustainedAPI(app=app)

    # ss.refresh("Tins, Pickles & Chutney")
    # TODO: At some point need to refresh all categories in below as for all before Tins, Pickles & Chutney, i only fetched the first page

    cats = ss.get_categories()
    cat_names = [c.name for c in cats]
    products = ss.get_products()

    cats_to_refresh = [
        "Baby Foods & Drinks",
        "Bacon & Sausages",
        "Beef",
        "Beer & Cider",
        "Biscuits",
        "Bread",
        "Breakfast Foods",
        "Butter & Margarine",
        "Cakes",
        "Carbonated Drinks",
        "Cereals",
        "Champagne & Rose",
        "Cheese",
        "Chicken & Turkey",
        "Chocolate & Sweets",
        "Cooking Sauces",
        "Crisps & Snacks",
        "Deli",
        "Dried Fruits & Nuts",
        "Eggs",
        "Fish & Seafood",
        "Frozen Foods",
        "Fruit Juice & Drinks",
        "Fruits",
        "Game",
        "Herbs",
        "Home Baking & Sugar",
        "Hot Beverages",
        "Jams & Spread",
        "Lamb",
        "Meat Free Alternatives",
        "Milk & Cream",
        "Organic Produce",
        "Other Alcoholic Beverages",
        "Other Fresh Meat & Fish",
        "Other Non-Alcoholic Drinks",
        "Pork",
        "Ready Made Foods",
        "Rice & Pasta",
        "Soups",
        "Spirit & Liqueurs",
        "Table Sauces & Condiments",
        "Tins",
        "Pickles & Chutney",
        "Veal & Venison",
        "Vegetables",
        "Water",
        "Wine",
        "World Foods",
        "Yoghurts",
    ]

    for c in cats_to_refresh:
        ss.refresh_products_lists(c)

    print("done")
    # with open('sustained.json', 'r') as f:
    #     localData = json.load(f)
    # flatten_products = []
    # for ps in localData['old_products']:
    #     flatten_products += ps
    # ids = [a['id'] for a in flatten_products]
    # for ps in localData['products']:
    #     flatten_products += [p for p in ps if p['id'] not in ids]
    # all_products = flatten_products
    # print(len(all_products))
    # for cat in localData['categories']:
    #     cat_products = [p for p in all_products if p['category'] == cat['name']]
    #     cat_id = cat['id']
    #     fn = f'localstorage/sustained-{cat_id}.json'

    #     if not os.path.exists(fn) and cat_products:
    #         with open(fn, 'w') as f:
    #             json.dump(cat_products, f, indent=2)
    #         print(f'wrote cat_products to "{fn}"')
    #     else:
    #         ss.refresh(cat['name'])
