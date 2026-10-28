"""Generate the demo knowledge base PDF for Acme (a fictional company).

Run from the project root:  python -m scripts.build_kb_pdf
Output: knowledge_base/acme_knowledge_base.pdf
Requires: pip install reportlab
"""

import argparse
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import ListFlowable, ListItem, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

DEFAULT_OUTPUT = Path("knowledge_base/acme_knowledge_base.pdf")
EFFECTIVE_DATE = "October 1, 2026"

# Each section: (heading, blocks). A block is a paragraph (str), a bullet list (list[str]),
# or a table (tuple of rows, first row is the header).
Block = str | list[str] | tuple[list[str], ...]

SECTIONS: list[tuple[str, list[Block]]] = [
    (
        "1. About Acme",
        [
            "Acme Commerce, Inc. (\"Acme\") is an online retailer that sells books and consumer electronics "
            "to customers in the United States and 40 other countries. Acme sells products directly "
            "(\"Sold by Acme\") and also hosts independent sellers on the Acme Marketplace (\"Sold by a "
            "Marketplace seller\"). The seller is shown on every product page and in every order confirmation.",
            "Our catalog includes print books, ebooks and audiobooks across all genres, and electronics such "
            "as tablets, e-readers, laptops, monitors, audio, smart home devices, streaming devices and "
            "accessories. Acme-brand devices (Lumina, Paperlight, Echo Mini, Stream Stick and Volt) are designed "
            "and supported by Acme.",
            "This document is the official reference for Acme customer policies. If anything in a product "
            "listing conflicts with this document, this document applies.",
        ],
    ),
    (
        "2. Acme Plus Membership",
        [
            "Acme Plus is a paid membership that costs $12.99 per month or $119 per year (US pricing, plus "
            "applicable tax). New members get a 30-day free trial. Benefits:",
            [
                "Free Two-Day Shipping on eligible items, with no minimum order.",
                "Free Same-Day Delivery on orders of $35 or more in eligible metro areas.",
                "Free return shipping on change-of-mind returns of eligible items.",
                "Acme Reads: borrow one ebook per month from a rotating catalog of over 2,000 titles.",
                "Early access (30 minutes) to Lightning Deals and member-only prices on Acme devices.",
            ],
            "You can cancel Acme Plus at any time from Account > Memberships. If you cancel during the free "
            "trial you are not charged. Annual members who cancel within 3 business days of being charged and "
            "have not used any benefit receive a full refund. After that, annual memberships are refunded "
            "pro rata for unused full months. Monthly memberships are not refunded for the current month.",
        ],
    ),
    (
        "3. Shipping and Delivery",
        [
            "Orders are processed within 1 business day. Delivery times start once an order ships and exclude "
            "weekends and US federal holidays unless stated otherwise.",
            (
                ["Option", "Delivery time", "Cost (non-members)", "Cost (Acme Plus)"],
                ["Standard", "4-6 business days", "Free over $35, otherwise $5.99", "Free"],
                ["Two-Day", "2 business days", "$9.99", "Free"],
                ["One-Day", "Next business day if ordered by 12 p.m. local time", "$14.99", "$4.99"],
                ["Same-Day", "Same day if ordered by 1 p.m. (eligible metros)", "Not available", "Free over $35, otherwise $2.99"],
                ["International", "7-14 business days", "From $14.99, shown at checkout", "From $14.99"],
            ),
            "Carriers: Acme ships with UPS, FedEx, USPS and Acme Logistics. A tracking number is emailed when "
            "the order ships and is also shown in Your Orders.",
            "International orders: import duties and taxes are estimated and collected at checkout, so there "
            "are no extra charges on delivery. Some electronics with lithium batteries cannot be shipped to "
            "certain countries; the product page shows if an item is not eligible.",
            "Pre-orders ship so that they arrive on or near the release date. You are charged when the "
            "pre-order ships, not when you place it.",
            "Missing packages: if tracking shows an order as delivered but you can't find it, check around "
            "your delivery location and with neighbors, then wait 48 hours, because carriers sometimes mark "
            "packages delivered early. If it still hasn't arrived, contact Acme Support and we will send a "
            "replacement or issue a full refund.",
        ],
    ),
    (
        "4. Changing or Cancelling an Order",
        [
            "You can cancel an order, or change its shipping address or delivery speed, while its status is "
            "Processing. Go to Your Orders and select Cancel items or Change address.",
            "Once an order has Shipped it can no longer be cancelled. You can refuse the delivery or return "
            "the item under the returns policy below. Cancelled orders are refunded to the original payment "
            "method, usually within 3-5 business days.",
            "Pre-orders can be cancelled at any time before they ship at no charge.",
        ],
    ),
    (
        "5. Returns and Refunds",
        [
            "Most items sold by Acme can be returned within 30 days of delivery for a full refund. Start a "
            "return from Your Orders > Return or replace items, choose a reason, and print the prepaid label "
            "or use a QR code at a drop-off point.",
            [
                "Books (print): returnable within 30 days in the condition received. Books with writing, "
                "highlighting or water damage that occurred after delivery are not eligible.",
                "Ebooks: purchases can be cancelled for a full refund within 7 days if less than 10% of the "
                "book has been read. Audiobooks can be exchanged within 365 days of purchase.",
                "Electronics: returnable within 30 days with all original accessories, manuals and packaging. "
                "Opened electronics returned without original accessories or with signs of use beyond "
                "inspection may be charged a restocking fee of up to 15% of the item price.",
                "Not returnable: digital content (except as above), downloadable software, gift cards, and "
                "items marked \"Final sale\" on the product page.",
                "Marketplace items: returns are handled by the seller under policies that are at least as "
                "generous as Acme's 30-day policy. The Acme Assurance Guarantee (section 9) applies if a "
                "seller doesn't respond.",
            ],
            "Return shipping is free if the item is damaged, defective or not what you ordered, and for Acme "
            "Plus members on eligible items. For other change-of-mind returns, $5.99 is deducted from the refund.",
            "Holiday returns: items delivered between November 1 and December 31 can be returned until January 31 "
            "of the following year.",
            "Refund timing: once the return is received and inspected (usually within 2 business days), refunds "
            "to Acme gift card balance arrive within 24 hours and refunds to credit or debit cards take 3-5 "
            "business days. The original shipping charge is refunded only if the return is due to an Acme error.",
        ],
    ),
    (
        "6. Damaged, Defective or Wrong Items",
        [
            "If an item arrives damaged, stops working, or is not what you ordered, report it within 30 days of "
            "delivery from Your Orders or by contacting Acme Support. Please keep the item and packaging and, if "
            "possible, take photos.",
            "You can choose a free replacement (if in stock) or a full refund, including the original shipping "
            "charge. Acme provides a prepaid return label. For large or heavy items, we may arrange a pickup or "
            "ask you to recycle the item instead of returning it.",
            "Damaged, defective and wrong-item cases are always handled by a human support specialist, who "
            "will contact you within 24 hours.",
            "After 30 days, defective electronics are covered by the warranty (section 7).",
        ],
    ),
    (
        "7. Warranty and Acme Protect",
        [
            "Acme-brand devices come with a 1-year Acme Limited Warranty from the delivery date. It covers defects "
            "in materials and workmanship under normal use. It does not cover accidental damage, liquid damage "
            "(except for devices rated water-resistant, within their rating), loss, theft or unauthorized repairs.",
            "Other brands are covered by the manufacturer's warranty, which is listed on the product page. Acme "
            "Support can help you contact the manufacturer.",
            "Acme Protect is an optional extended protection plan for electronics, available at checkout or within "
            "30 days of purchase:",
            (
                ["Plan", "Coverage", "Price"],
                ["Acme Protect 2-Year", "Defects after the warranty ends + accidental drops and spills", "From $19.99"],
                ["Acme Protect 3-Year", "Same as 2-Year, for 3 years from purchase", "From $29.99"],
            ),
            "To make a warranty or Acme Protect claim, go to Your Orders > Get product support. Approved claims are "
            "repaired, replaced with the same or a comparable model, or refunded. Acme Protect has no deductible "
            "and covers up to 2 accidental-damage claims per plan.",
        ],
    ),
    (
        "8. Prices, Price Adjustments and Promotions",
        [
            "Prices are shown in your local currency and include any applicable discounts. Sales tax is calculated "
            "at checkout based on the delivery address.",
            "Price adjustment: if the price of an item sold by Acme drops within 7 days of delivery, you can request "
            "a refund of the difference through Acme Support. This does not apply to Marketplace items, Lightning "
            "Deals, coupons, or prices from other retailers. Acme does not match competitor prices.",
            "Pre-order price guarantee: if the price of a pre-ordered item drops between the time you order and the "
            "release date, you are charged the lowest price.",
            "Promotional codes must be applied at checkout, cannot be combined unless stated, and have no cash value.",
        ],
    ),
    (
        "9. Acme Assurance Guarantee (Marketplace Purchases)",
        [
            "The Acme Assurance Guarantee protects purchases from Marketplace sellers. You can file a claim if:",
            [
                "The item has not arrived 3 days after the latest estimated delivery date, or",
                "The item is damaged, defective, materially different from the listing, or",
                "The seller has not issued an agreed refund.",
            ],
            "First contact the seller from Your Orders and allow 48 hours for a reply. If the issue isn't resolved, "
            "file a claim within 90 days of the latest estimated delivery date. Acme reviews claims within 7 days "
            "and covers up to $2,500 per order.",
        ],
    ),
    (
        "10. Payment Methods",
        [
            [
                "Credit and debit cards: Visa, Mastercard, American Express and Discover.",
                "PayPal and Acme gift cards. Gift card balance is applied before other payment methods.",
                "Acme Pay Later: split purchases of $150 or more into 4 interest-free payments over 6 weeks, or 6 "
                "or 12 monthly payments, subject to approval. Rates and terms are shown at checkout.",
            ],
            "Your payment method is authorized when you place the order and charged when the items ship. If an "
            "order ships in several packages, you may see several charges.",
            "Acme will never ask for your password, full card number or one-time passcodes by phone, chat or email.",
        ],
    ),
    (
        "11. Books at Acme",
        [
            "Formats: most titles are available as hardcover, paperback, ebook and audiobook, and each format has "
            "its own price and product page. Ebooks are read in the free Acme Reader app or on Paperlight "
            "e-readers and Lumina tablets.",
            "Used books from Marketplace sellers are graded Like New, Very Good, Good or Acceptable. The grade and "
            "the seller's notes are shown on the listing.",
            "Signed and limited editions are final sale unless they arrive damaged or are not as described.",
            "Bulk orders of 25 or more copies of one title (for schools, book clubs or businesses) can get a volume "
            "discount through Acme Business Sales.",
        ],
    ),
    (
        "12. Electronics at Acme",
        [
            "Compatibility: check the product page specifications, such as ports, operating system and power, "
            "before buying. Acme Support can confirm only the specifications listed on the product page.",
            "Power: devices sold on the US store include US plugs and are designed for 100-240V unless the product "
            "page says otherwise.",
            "Acme Trade-In: eligible used devices (tablets, e-readers, smart speakers, streaming devices and "
            "laptops) can be traded in for an Acme gift card. Get a quote from the Trade-In page; shipping is free.",
            "Recycling: Acme recycles old electronics and batteries for free. Request a free shipping label from "
            "the Recycling page.",
        ],
    ),
    (
        "13. Account, Privacy and Security",
        [
            "To discuss a specific order, Acme Support may ask you to confirm the email address on the account. "
            "Support agents can see order details but never your full payment card number or password.",
            "You can download or delete your personal data from Account > Privacy. Deleting your account cancels "
            "Acme Plus and removes access to purchased ebooks and audiobooks.",
            "If you notice an order you didn't place, contact Acme Support immediately. These cases are always "
            "handled by a human specialist.",
        ],
    ),
    (
        "14. Contacting Acme Support",
        [
            [
                "Chat: 24 hours a day, 7 days a week, from the Help page or the Acme app. The AI assistant answers "
                "first and can transfer you to a specialist at any time.",
                "Phone: 1-800-555-0199, 6 a.m. to 10 p.m. Eastern Time, every day.",
                "Email: support@acme.example. We reply within 24 hours.",
            ],
            "The following are always handled by a human specialist: payment disputes and chargebacks, damaged or "
            "defective items, suspected fraud or unauthorized orders, and any customer who asks for a person. "
            "Typical response time for escalated chats is under 10 minutes during phone support hours.",
        ],
    ),
    (
        "15. Frequently Asked Questions",
        [
            "<b>Can I change my delivery address after ordering?</b> Yes, while the order status is Processing. "
            "After it ships, you can ask the carrier to redirect the package using the tracking number.",
            "<b>When will I be charged for a pre-order?</b> When it ships. You pay the lowest price offered between "
            "ordering and release.",
            "<b>Do I need the original box to return electronics?</b> The original packaging and all accessories "
            "are required for a full refund. Without them, a restocking fee of up to 15% may apply.",
            "<b>Is return shipping free?</b> It is free for damaged, defective or wrong items and for Acme Plus "
            "members. Otherwise $5.99 is deducted from the refund.",
            "<b>How long do refunds take?</b> Gift card refunds arrive within 24 hours and card refunds within "
            "3-5 business days after we receive and inspect the return.",
            "<b>Does Acme ship internationally?</b> Yes, to 40 countries. Duties and taxes are collected at checkout.",
        ],
    ),
]


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("title", parent=base["Title"], fontSize=22, spaceAfter=6),
        "subtitle": ParagraphStyle("subtitle", parent=base["Normal"], fontSize=10, textColor=colors.grey, alignment=1),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], spaceBefore=14, spaceAfter=6),
        "body": ParagraphStyle("body", parent=base["BodyText"], fontSize=10, leading=14, spaceAfter=6),
        "cell": ParagraphStyle("cell", parent=base["BodyText"], fontSize=9, leading=11),
    }


def _table(rows: tuple[list[str], ...], style: ParagraphStyle) -> Table:
    data = [[Paragraph(f"<b>{c}</b>" if i == 0 else c, style) for c in row] for i, row in enumerate(rows)]
    table = Table(data, repeatRows=1, hAlign="LEFT", colWidths=[(6.5 * inch) / len(rows[0])] * len(rows[0]))
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF7")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B0B8C4")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def build_pdf(output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    styles = _styles()
    doc = SimpleDocTemplate(
        str(output),
        pagesize=LETTER,
        leftMargin=inch,
        rightMargin=inch,
        topMargin=0.9 * inch,
        bottomMargin=0.9 * inch,
        title="Acme Customer Policies and Help Center",
        author="Acme Commerce, Inc. (fictional)",
    )

    story = [
        Paragraph("Acme Customer Policies and Help Center", styles["title"]),
        Paragraph(
            f"Books and Electronics | Effective {EFFECTIVE_DATE} | Fictional company, for demo purposes only",
            styles["subtitle"],
        ),
        Spacer(1, 0.25 * inch),
    ]
    for i, (heading, blocks) in enumerate(SECTIONS):
        if i == 7:  # keep the two halves of the document roughly balanced
            story.append(PageBreak())
        story.append(Paragraph(heading, styles["h2"]))
        for block in blocks:
            if isinstance(block, str):
                story.append(Paragraph(block, styles["body"]))
            elif isinstance(block, list):
                story.append(
                    ListFlowable(
                        [ListItem(Paragraph(item, styles["body"]), leftIndent=12) for item in block],
                        bulletType="bullet",
                        start="•",
                        leftIndent=14,
                    )
                )
            else:
                story += [_table(block, styles["cell"]), Spacer(1, 6)]

    def footer(canvas, document) -> None:
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.grey)
        canvas.drawString(inch, 0.5 * inch, "Acme Customer Policies and Help Center")
        canvas.drawRightString(LETTER[0] - inch, 0.5 * inch, f"Page {document.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    path = build_pdf(parser.parse_args().output)
    print(f"Wrote {path}")
