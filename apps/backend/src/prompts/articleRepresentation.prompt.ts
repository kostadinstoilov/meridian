export function getArticleRepresentationPrompt(title: string, url: string, text: string) {
  return `
Transform article into standardized format. No repeated info across fields.

Fields:
Topic: technology/politics/business/health/agriculture/sports/international
Subtopic: specific area (ai-research, elections, trade-policy)  
Geography: global/us/china/europe/[city]/[region] 
Scope: policy/technical/market/social-impact/breaking-news/analysis
Urgency: breaking/developing/routine/historical
Source: mainstream/trade/academic/government/blog
Entities: [max 5 key people/orgs/products/places]
Tags: [max 5 additional specifics not covered above]

Examples:

INPUT: """
Nvidia CEO Jensen Huang Warns Companies to Adopt AI Now

Nvidia CEO delivered stark warning to business leaders yesterday, stating companies must integrate AI immediately or face obsolescence. Speaking to Fortune 500 executives, emphasized current AI revolution represents 'once-in-a-lifetime transformation'. Stock surged 180% this year as AI chip demand accelerates.
"""

OUTPUT:
Topic: technology
Subtopic: business-strategy
Geography: us
Scope: market
Urgency: routine
Source: mainstream
Entities: [Jensen Huang, Nvidia, Fortune 500]
Tags: [stock-surge, 180-percent, chip-demand]

INPUT: """
Breaking: Emergency Wheat Export Ban by Inner Mongolia Agricultural Ministry

Ministry announced immediate wheat export suspension today, citing food security concerns amid drought. Affects 2.3 million tons scheduled for neighboring provinces. Farmers concerned about revenue losses, traders predict price volatility.
"""

OUTPUT:
Topic: agriculture
Subtopic: trade-policy
Geography: inner-mongolia
Scope: breaking-news
Urgency: breaking
Source: mainstream
Entities: [Inner Mongolia Agricultural Ministry]
Tags: [export-ban, drought, 2.3-million-tons, price-volatility]

INPUT: """"
# (${title})[${url}]

${text.slice(0, 1500)}...
"""

OUTPUT:
`.trim();
}
