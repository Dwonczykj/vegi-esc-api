def select_products_sql(productId:int):
  _sql = '''with products as (
SELECT 
  `product`.`createdAt` AS product_createdAt,
  `product`.`updatedAt` AS product_updatedAt,
  `product`.`id` AS product_id,
  `product`.`name` AS product_name,
  `product`.`description` AS product_description,
  `product`.`shortDescription` AS product_shortDescription,
  `product`.`basePrice` AS product_basePrice,
  `product`.`imageUrl` AS product_imageUrl,
  `product`.`isAvailable` AS product_isAvailable,
  `product`.`priority` AS product_priority,
  `product`.`vendor` AS product_vendor,
  `product`.`category` AS product_category,
  `product`.`isFeatured` AS product_isFeatured,
  `product`.`status` AS product_status,
  `product`.`ingredients` AS product_ingredients,
  `productcategory`.`createdAt` AS productcategory_createdAt,
  `productcategory`.`updatedAt` AS productcategory_updatedAt,
  `productcategory`.`id` AS productcategory_id,
  `productcategory`.`name` AS productcategory_name,
  `productcategory`.`imageUrl` AS productcategory_imageUrl,
  `productcategory`.`vendor` AS productcategory_vendor,
  `productcategory`.`categoryGroup` AS productcategory_categoryGroup,
  `categorygroup`.`id` AS categorygroup_id,
  `categorygroup`.`name` AS categorygroup_name,
  `categorygroup`.`imageUrl` AS categorygroup_imageUrl,
  `categorygroup`.`forRestaurantItem` AS categorygroup_forRestaurantItem,
  `productoption`.`createdAt` AS productoption_createdAt,
  `productoption`.`updatedAt` AS productoption_updatedAt,
  `productoption`.`id` AS productoption_id,
  `productoption`.`name` AS productoption_name,
  `productoption`.`product` AS productoption_product,
  `productoption`.`isRequired` AS productoption_isRequired,
  `productoptionvalue`.`createdAt` AS productoptionvalue_createdAt,
  `productoptionvalue`.`updatedAt` AS productoptionvalue_updatedAt,
  `productoptionvalue`.`id` AS productoptionvalue_id,
  `productoptionvalue`.`name` AS productoptionvalue_name,
  `productoptionvalue`.`description` AS productoptionvalue_description,
  `productoptionvalue`.`priceModifier` AS productoptionvalue_priceModifier,
  `productoptionvalue`.`isAvailable` AS productoptionvalue_isAvailable,
  `productoptionvalue`.`stockCount` AS productoptionvalue_stockCount,
  `productoptionvalue`.`brandName` AS productoptionvalue_brandName,
  `productoptionvalue`.`taxGroup` AS productoptionvalue_taxGroup,
  `productoptionvalue`.`stockUnitsPerProduct` AS productoptionvalue_stockUnitsPerProduct,
  `productoptionvalue`.`sizeInnerUnitValue` AS productoptionvalue_sizeInnerUnitValue,
  `productoptionvalue`.`sizeInnerUnitType` AS productoptionvalue_sizeInnerUnitType,
  `productoptionvalue`.`productBarCode` AS productoptionvalue_productBarCode,
  `productoptionvalue`.`supplier` AS productoptionvalue_supplier,
  `productoptionvalue`.`option` AS productoptionvalue_option
  FROM `vegi`.`product` product
	left join `vegi`.`productoption` productoption on product.id = productoption.product 
	left join `vegi`.`productoptionvalue` productoptionvalue on productoptionvalue.option = productoption.id 
	left join `vegi`.`productcategory` productcategory on productcategory.id = product.category 
	left join `vegi`.`categorygroup` categorygroup on categorygroup.id = productcategory.categoryGroup
  where product.id = @productId
)
,esc as (
	select
		products.*,
	  `escrating`.`createdAt` AS escrating_createdAt,
	  `escrating`.`updatedAt` AS escrating_updatedAt,
	  `escrating`.`id` AS escrating_id,
	  `escrating`.`productPublicId` AS escrating_productPublicId,
	  `escrating`.`rating` AS escrating_rating,
	  `escrating`.`evidence` AS escrating_evidence,
	  `escrating`.`calculatedOn` AS escrating_calculatedOn,
	  `escrating`.`product` AS escrating_product,
	  `escexplanation`.`createdAt` AS escexplanation_createdAt,
	  `escexplanation`.`updatedAt` AS escexplanation_updatedAt,
	  `escexplanation`.`id` AS escexplanation_id,
	  `escexplanation`.`title` AS escexplanation_title,
	  `escexplanation`.`description` AS escexplanation_description,
	  `escexplanation`.`measure` AS escexplanation_measure,
	  `escexplanation`.`escrating` AS escexplanation_escrating
  FROM products
    left join `vegi`.`escrating` escrating on products.product_id = `escrating`.`product`
	left join `vegi`.`escexplanation` escexplanation on escrating.id = escexplanation.escrating
    where (1=1)
        and (TIMESTAMPDIFF(HOUR, FROM_UNIXTIME(`escrating`.`createdAt`), UNIX_TIMESTAMP(NOW())) <= 24 or `escrating`.`id` is null)
)
select * from esc order by esc.product_id, esc.escrating_createdAt DESC
'''
  _sql = _sql.replace('@productId',f'{productId}')
  return _sql