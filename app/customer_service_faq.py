# app/customer_service_faq.py
# Knowledge base for customer_service_agent. Interim source of truth until
# support content moves to a retrieval backend — edit here to update answers.

CUSTOMER_SERVICE_FAQ = """
HipTraveler FAQ & Support

## Using the App
Q: How does HipTraveler help me plan my trip?
A: HipTraveler combines AI-powered planning with curated insights from the HipTraveler community, travel bloggers, local experts, and travel agents. You can filter by destination, experiences, and travel style, and instantly generate itineraries for most worldwide destinations.
Q: Can I save and edit my trips?
A: Yes. All trips can be saved to your profile. You can update dates, swap activities, add/remove days and fully personalize your itinerary.
Q: Do I need to download the app to use HipTraveler?
A: No. You can build trips, browse itineraries and travel content on the web, but the app provides the full experience including offline access, Google Maps downloads, real time "AroundMe" functionality that helps you to find places of interest that match your travel style, global eSIM connectivity, remote access to Travel Agents, and soon, Travelogging features that help you to document your trip in real time.
Q: What devices does HipTraveler support?
A: HipTraveler is available on iOS, Android, and web. Your account syncs across devices on all platforms.
Q: How do I manage my account (reset password, delete account, etc.)?
A: You can reset your password or update your email directly in the profile settings. To permanently delete your account, submit a request through the app or email support@hiptraveler.com.

## Premium Membership
Q: What's included in HipTraveler Premium?
A: Premium unlocks global eSIM connectivity, access to our Travel Concierge, and the ability to export trips directly to Google Maps.
Q: How does the eSIM work?
A: Premium includes 20 GB of data valid for 365 days across 140 countries. Top-ups can be purchased anytime through the app.
Q: What is the Travel Concierge?
A: The Travel Concierge connects you to real travel agents who can help with complex itineraries, personal recommendations, and special requests.
Q: Can I export trips to Google Maps?
A: Yes. Premium members can export itineraries directly to Google Maps for navigation and real-time directions.
Q: What are the Premium subscription options (monthly vs. annual)?
A: You can choose monthly or annual billing. Plans are managed through Stripe for secure payments.
Q: How do I cancel my subscription?
A: Subscriptions can be canceled anytime in the app. You'll continue to enjoy Premium until the end of your billing cycle.
Q: How much data is included with the eSIM, and can I top up?
A: Premium comes with 20 GB of data per year. If you need more, top-ups are available for purchase in-app.

## Travelogging & Content
Q: What is Travelogging?
A: Travelogging allows you to capture your journeys with stories, photos, and hacks. It's your digital travel journal, stored in your profile. You can create Travelogues in the web and mobile versions of HipTraveler.
Q: Can I share my Travelogues with others?
A: Yes. You can publish Travelogues to the HipTraveler community, share to social channels, or copy/paste a link to share directly.
Q: How do I share Travelogues to social media?
A: Each Travelogue can be shared via a generated link or shared directly to social channels like Instagram, Facebook, or Twitter.
Q: How does HipTraveler moderate community content?
A: Our moderators review Travel Hacks and Travelogues regularly. Users can flag inappropriate content, and depending on severity, accounts may be suspended or removed.
Q: What are Dream Trips?
A: Dream Trips are future trips you highlight in your profile. They can be used for inspiration or planning when you're ready to book. Also, when you mark a trip as a Dream Trip, download our app, and register for our sweepstakes, you become eligible to win your Dream Trip.
Q: Can I export Travelogues to WordPress?
A: Not yet. Exporting to WordPress will be available in 2026.

## Booking & Payments
Q: Is it safe to book through HipTraveler?
A: Yes. All payments are processed securely through Stripe. Your personal and financial data is encrypted and protected.
Q: What payment options are supported?
A: We accept major credit/debit cards, Apple Pay, Google Pay, and regional payment options depending on your country.
Q: What currencies are supported?
A: HipTraveler supports multiple currencies, displayed based on your location. Prices are converted using Stripe's exchange rates.
Q: Can I use loyalty points on bookings?
A: Yes. Loyalty points can be applied toward future bookings, with discounts reflected at checkout.
Q: Will I be charged immediately when I book?
A: This depends on the property or provider. Some bookings require immediate payment; others allow pay-at-property options. The policy is displayed before checkout.
Q: Does HipTraveler charge booking or reservation fees?
A: No. HipTraveler does not charge extra booking fees.
Q: Are taxes included in the price?
A: Taxes and fees are displayed before you confirm your booking.
Q: Do I need to pay a deposit at the property?
A: Some properties may require a deposit on arrival. Details are included in your booking confirmation.
Q: How do refunds work, and how long do they take?
A: Refunds are processed through Stripe. Most appear within 5–10 business days, depending on your bank.
Q: Why was my payment declined?
A: Payments may fail due to incorrect details, insufficient funds, or card restrictions. Try another method or contact your bank.
Q: What does the crossed-out rate next to my room type mean?
A: This shows the original price before discounts or promotions were applied.

## Cancellations
Q: Can I cancel my booking?
A: Yes. Cancellations are subject to the property's policy, which is shown during booking and in your confirmation email.
Q: If I need to cancel my booking, will I pay a fee?
A: If your booking includes free cancellation, you won't pay a fee. Otherwise, the property may charge based on their policy.
Q: Can I cancel or change my dates for a non-refundable booking?
A: Non-refundable bookings usually incur a charge. In some cases, you may request free cancellation through the app, but approval is at the property's discretion.
Q: How do I know if my booking was canceled?
A: You'll receive a confirmation email. If you don't see it within 24 hours, check spam or contact the property directly.
Q: Where can I find my cancellation policy?
A: It's included in your booking confirmation and displayed before checkout.
Q: Where can I see the payment policy for my booking?
A: Payment policies are listed in your booking confirmation under the pricing section.
Q: How do I request a refund for a canceled booking?
A: You can request a refund directly in the app or within the My Trips section of your profile page. Our support team will work with the property to process it.

## Pricing
Q: Is breakfast included in the price?
A: If breakfast is included, it will be shown under room details before booking.
Q: What does the price include?
A: Prices include the listed room, amenities, and services. Any extras will be detailed before checkout.
Q: Are the prices shown per person or per room?
A: Unless specified otherwise, prices are per room.
Q: Are there any extra fees I should know about?
A: Some properties may add resort fees, cleaning fees, or local taxes. HipTraveler makes our best effort to always display these additional fees before you confirm your booking.
Q: Do I pay the full price for my child?
A: Child pricing varies by property and will be displayed during booking.
Q: Can I use discount coupons?
A: Currently, HipTraveler does not accept third-party coupons.
Q: Does HipTraveler offer discounts with loyalty cards or memberships?
A: Not yet. We are exploring partnerships with airlines and hotel loyalty programs.

## Check-In & Stay Information
Q: What time is check-in and check-out?
A: Times vary by property and are listed in your booking confirmation.
Q: What ID or documents do I need to check in?
A: Most properties require a government-issued photo ID and your booking confirmation. For locations outside of your home country, often times, the hotel will request a copy of your passport as well.
Q: Can I request an early check-in or late check-out?
A: Yes, requests can be made through the app. Approval depends on property availability.
Q: How do I make special requests (extra bed, accessibility needs, etc.)?
A: You can add requests during booking or by contacting the property directly.
Q: Are pets allowed at the property?
A: Pet policies vary. Details are included in the property's listing.
Q: Is parking included?
A: Parking details are shown in the property's amenities section. Some may charge extra.
Q: How do I know if a property is accessible?
A: Accessibility features are listed under amenities. If unclear, contact the property directly.

## Refunds & Policies
Q: How long does it take to receive my refund?
A: Refunds usually appear within 5–10 business days, though some banks take longer.
Q: Who issues the refund — HipTraveler or the property?
A: Refunds are processed through HipTraveler and Stripe, in coordination with the property.
Q: Are security deposits refundable?
A: Yes. Deposits are typically refunded by the property after your stay, provided no damage occurs.
Q: Why is my refund amount different from what I paid?
A: Exchange rates, bank fees, or partial refund policies may affect the final amount.
Q: What happens if my refund doesn't arrive?
A: Contact support@hiptraveler.com with your booking ID. We'll help track it.

## Account & Security
Q: How do I reset my password?
A: Tap "Forgot Password" on the login screen to receive a reset link by email.
Q: How do I change my email address?
A: You can update your email in profile settings.
Q: How do I delete my account permanently?
A: Submit a request in-app or email support@hiptraveler.com. Once deleted, your data cannot be recovered.
Q: How does HipTraveler protect my personal information?
A: We use encryption and industry-standard security protocols to protect your data.
Q: Can I enable two-factor authentication?
A: Currently, 2FA is not available but is planned for future updates.

## Safety & Trust
Q: How does HipTraveler verify properties and experiences?
A: We work with trusted providers and booking partners. We make our best effort to verify properties before they are listed.
Q: What should I do if I suspect a scam or fraudulent listing?
A: Report it through the app or email support@hiptraveler.com immediately.
Q: How does HipTraveler protect travelers while abroad?
A: We provide secure booking, data protection, and access to real support if issues arise during your trip.
Q: How can I report inappropriate behavior or content?
A: Use the "Report" option on any post or contact support.

## Loyalty & Rewards
Q: How do I earn loyalty points?
A: Points are earned on bookings made through HipTraveler, sharing content, creating and sharing content on the platform, and referring HipTraveler to others.
Q: How do I redeem points on bookings?
A: Points can be applied during checkout as a discount. If you're logged in at checkout, you will see your available points and you can then choose to redeem.
Q: Do loyalty points expire?
A: Yes, points expire 24 months after they are earned.
Q: Are there different reward tiers?
A: Not yet, but tiered rewards are in development.

## Contact & Communication
Q: How do I contact HipTraveler support?
A: Email us at support@hiptraveler.com or use the in-app Support section.
Q: How quickly will support respond?
A: We aim to respond within 3 hours during support hours.
Q: How can I contact the property directly?
A: Property contact details are included in your booking confirmation.
Q: What should I do if I don't receive my booking confirmation?
A: Check your spam folder. If you don't see it within 24 hours, contact support.
Q: What email should I use for support?
A: Use support@hiptraveler.com for all inquiries.

## Software Support & Bug Resolution
Q: How can I report a technical issue or bug?
A: Use the in-app Support section or email support@hiptraveler.com with screenshots and device details.
Q: What happens after I report a bug?
A: Our support team will confirm within ~3 hours and keep you updated through your dashboard.
Q: How long does it take to resolve bugs?
A: Minor issues may be fixed in days. Complex problems may take longer, depending on third-party involvement.
Q: What if I encounter an integration issue (e.g., Google Maps export)?
A: Integration bugs are logged as high priority and resolved in coordination with partners.
Q: How can I track known issues?
A: You can view ongoing issues and updates on our public-facing status page.
Q: Should I reinstall the app if I'm having issues?
A: Yes, reinstalling can resolve many issues. If problems continue, contact support.
""".strip()
