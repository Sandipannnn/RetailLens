// Display aliases only: the source dataset does not include real store/product names.
// Keep these ordered by dataset ID so selections still request the original series.
const stores = [
  'Central Market', 'Riverside Market', 'Parkside Market', 'Hillcrest Market',
  'Lakeside Market', 'Oakwood Market', 'West End Market', 'Eastgate Market',
  'Northgate Market', 'Southgate Market',
]

const products = [
  'Basmati Rice', 'Whole Wheat Flour', 'Granulated Sugar', 'Table Salt', 'Sunflower Oil',
  'Olive Oil', 'Whole Milk', 'Plain Yogurt', 'Salted Butter', 'Cheddar Cheese',
  'Eggs', 'White Bread', 'Whole Wheat Bread', 'Rolled Oats', 'Corn Flakes',
  'Penne Pasta', 'Instant Noodles', 'Tomato Ketchup', 'Mayonnaise', 'Peanut Butter',
  'Strawberry Jam', 'Honey', 'Ground Coffee', 'Black Tea', 'Green Tea',
  'Orange Juice', 'Apple Juice', 'Bottled Water', 'Cola', 'Sparkling Water',
  'Potato Chips', 'Cream Crackers', 'Chocolate Cookies', 'Milk Chocolate', 'Roasted Almonds',
  'Cashews', 'Raisins', 'Red Lentils', 'Chickpeas', 'Kidney Beans',
  'Canned Tomatoes', 'Sweet Corn', 'Frozen Peas', 'Frozen Mixed Vegetables', 'Vanilla Ice Cream',
  'Dishwashing Liquid', 'Laundry Detergent', 'Hand Soap', 'Paper Towels', 'Toothpaste',
]

export function storeName(id: number | null): string {
  return id === null ? '—' : stores[id - 1] ?? `Unknown store (${id})`
}

export function itemName(id: number | null): string {
  return id === null ? '—' : products[id - 1] ?? `Unknown product (${id})`
}
