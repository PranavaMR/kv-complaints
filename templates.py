"""
SYNTHETIC complaint templates (2 per aspect) and neutral length fillers.
Fillers are appended to match the restaurant's median review length; they must contain no lexicon term.
"""

TEMPLATES = {
    "access": [
        "I was really looking forward to this place, but getting inside was a nightmare. "
        "My mother uses a wheelchair and there is no step-free way in: three steps at the front "
        "door and no ramp anywhere. A server told us the side door was locked and only staff had the key. "
        "We ended up carrying her chair up the steps while other diners watched, which was humiliating for her.",
        "Food was fine, but I can't recommend it to anyone with mobility issues. "
        "I'm in a wheelchair and the only entrance has a steep step with no ramp. "
        "When I asked about accessibility, the host shrugged and said they get asked a lot. "
        "The restroom was also down a narrow flight of stairs, so I had to leave early.",
    ],
    "double_charge": [
        "Check your bank statement after eating here. They double charged my card for our dinner, "
        "the exact same amount went through twice within a minute. "
        "I called the next day and the manager said it was a system glitch and would fix itself. "
        "A week later it still hadn't been refunded and I had to dispute it with my bank.",
        "Be careful paying by card. I was charged twice for a single meal, two identical charges "
        "on my credit card the same evening. The staff insisted their machine never makes mistakes "
        "until I showed them my banking app. It took three phone calls and ten days to get the "
        "duplicate charge reversed.",
    ],
    "towed": [
        "We parked in the restaurant's own lot, which had no signs about time limits. "
        "When we came out after dinner my car had been towed. The tow company said the restaurant "
        "had called them because we stayed past ninety minutes. Nobody inside warned us, and getting "
        "the car back cost us over two hundred dollars and a cab ride across town.",
        "Do not park in their lot. My car was towed while we were still eating dessert, "
        "even though we were paying customers. The staff said the lot is shared and the tow truck "
        "comes by every night. There was one tiny sign hidden behind a bush. "
        "That ruined what was otherwise an okay evening.",
    ],
}

FILLERS = [
    "The fries were hot and the portion sizes were decent.",
    "Our server was friendly enough when she finally came over.",
    "The dining room was loud on a Friday night.",
    "Drinks took about fifteen minutes to arrive.",
    "The burger was cooked the way I asked, a little salty.",
    "Dessert was the best part of the meal, a warm brownie with ice cream.",
    "Prices are about what you would expect for the area.",
    "The menu has a lot of choices, maybe too many.",
    "We waited around twenty minutes for a table even with a reservation.",
    "The salad was fresh but the dressing was bland.",
    "Music was a bit too loud to hold a conversation.",
    "I might come back for lunch, but not anytime soon.",
]
