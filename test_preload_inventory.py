"""
Inventory preload: every detected location must become one
POST /trip-planner/preloadInventory with the contract's formattedAddr shape.

Pure unit tests — no server, no network. Google Geocoding and the backend are
replaced with an httpx MockTransport.

Run:  python test_preload_inventory.py      (or: python -m pytest test_preload_inventory.py)
"""
import asyncio
import json
import sys
import unittest

import httpx

from app import preload_inventory as pi
from app.config import settings


# ─── Geocode fixtures (trimmed real Google Geocoding shapes) ──────────

def _c(long, short, *types):
    return {"long_name": long, "short_name": short, "types": list(types)}


SAN_FRANCISCO = {
    "types": ["locality", "political"],
    "formatted_address": "San Francisco, CA, USA",
    "address_components": [
        _c("San Francisco", "SF", "locality", "political"),
        _c("San Francisco County", "San Francisco County", "administrative_area_level_2", "political"),
        _c("California", "CA", "administrative_area_level_1", "political"),
        _c("United States", "US", "country", "political"),
    ],
}

KARACHI = {
    "types": ["locality", "political"],
    "formatted_address": "Karachi, Pakistan",
    "address_components": [
        _c("Karachi", "Karachi", "locality", "political"),
        _c("Karachi Division", "Karachi Division", "administrative_area_level_2", "political"),
        _c("Sindh", "Sindh", "administrative_area_level_1", "political"),
        _c("Pakistan", "PK", "country", "political"),
    ],
}

TOKYO = {
    "types": ["administrative_area_level_1", "political"],
    "formatted_address": "Tokyo, Japan",
    "address_components": [
        _c("Tokyo", "Tokyo", "administrative_area_level_1", "political"),
        _c("Japan", "JP", "country", "political"),
    ],
}

SINGAPORE = {
    "types": ["country", "locality", "political"],
    "formatted_address": "Singapore",
    "address_components": [
        _c("Singapore", "SG", "country", "political"),
        _c("Singapore", "Singapore", "locality", "political"),
    ],
}

JAPAN = {
    "types": ["country", "political"],
    "formatted_address": "Japan",
    "address_components": [_c("Japan", "JP", "country", "political")],
}

EUROPE = {
    "types": ["continent", "establishment"],
    "formatted_address": "Europe",
    "address_components": [_c("Europe", "Europe", "continent", "establishment")],
}

BARCELONA = {
    "types": ["locality", "political"],
    "formatted_address": "Barcelona, Spain",
    "address_components": [
        _c("Barcelona", "Barcelona", "locality", "political"),
        _c("Barcelona", "B", "administrative_area_level_2", "political"),
        _c("Catalonia", "CT", "administrative_area_level_1", "political"),
        _c("Spain", "ES", "country", "political"),
    ],
}

MADRID = {
    "types": ["locality", "political"],
    "formatted_address": "Madrid, Spain",
    "address_components": [
        _c("Madrid", "Madrid", "locality", "political"),
        _c("Community of Madrid", "MD", "administrative_area_level_1", "political"),
        _c("Spain", "ES", "country", "political"),
    ],
}

CALIFORNIA = {
    "types": ["administrative_area_level_1", "political"],
    "formatted_address": "California, USA",
    "address_components": [
        _c("California", "CA", "administrative_area_level_1", "political"),
        _c("United States", "US", "country", "political"),
    ],
}

FRANCE = {
    "types": ["country", "political"],
    "formatted_address": "France",
    "address_components": [_c("France", "FR", "country", "political")],
}

PARIS = {
    "types": ["locality", "political"],
    "formatted_address": "Paris, France",
    "address_components": [
        _c("Paris", "Paris", "locality", "political"),
        _c("Île-de-France", "IDF", "administrative_area_level_1", "political"),
        _c("France", "FR", "country", "political"),
    ],
}

GEOCODE_BY_NAME = {
    "san francisco": SAN_FRANCISCO,
    "karachi": KARACHI,
    "tokyo": TOKYO,
    "singapore": SINGAPORE,
    "japan": JAPAN,
    "europe": EUROPE,
    "barcelona": BARCELONA,
    "madrid": MADRID,
    "california": CALIFORNIA,
    "ca": CALIFORNIA,
    "france": FRANCE,
    "paris": PARIS,
}


class SplitTests(unittest.TestCase):
    """A multi-city explore `location` string is split into its cities."""

    def test_separators(self):
        self.assertEqual(pi.split_location_names("Barcelona, Madrid"), ["Barcelona", "Madrid"])
        self.assertEqual(pi.split_location_names("Barcelona and Madrid"), ["Barcelona", "Madrid"])
        self.assertEqual(pi.split_location_names("Paris / London & Rome"), ["Paris", "London", "Rome"])
        self.assertEqual(pi.split_location_names("Tokyo"), ["Tokyo"])
        self.assertEqual(pi.split_location_names(""), [])

    def test_and_is_a_separator(self):
        # "and" splits, so a country whose name contains it splits too. Both
        # halves geocode to that same country and are skipped as country-level,
        # so nothing wrong is preloaded.
        self.assertEqual(pi.split_location_names("Trinidad and Tobago"), ["Trinidad", "Tobago"])
        self.assertEqual(pi.split_location_names("Sandton"), ["Sandton"])  # "and" inside a word is not a separator

    def test_duplicates_dropped_order_kept(self):
        self.assertEqual(pi.split_location_names("Madrid, Barcelona, madrid"), ["Madrid", "Barcelona"])

    def test_state_next_to_its_city_is_dropped(self):
        sf = pi.build_location(SAN_FRANCISCO)
        ca = pi.build_location(CALIFORNIA)
        self.assertEqual(pi.drop_parents([sf, ca]), [sf])

    def test_two_real_cities_both_kept(self):
        bcn = pi.build_location(BARCELONA)
        mad = pi.build_location(MADRID)
        self.assertEqual(pi.drop_parents([bcn, mad]), [bcn, mad])


class BuildLocationTests(unittest.TestCase):
    """formattedAddr / cityType / countryCode derivation from a geocode result."""

    def test_us_city_is_city_state_code_country_name(self):
        loc = pi.build_location(SAN_FRANCISCO)
        self.assertEqual(loc.formatted_addr, "San Francisco, CA, United States")
        self.assertEqual(loc.city_type, "locality")
        self.assertEqual(loc.country_code, "US")

    def test_non_us_city_is_city_country_name(self):
        loc = pi.build_location(KARACHI)
        self.assertEqual(loc.formatted_addr, "Karachi, Pakistan")
        self.assertEqual(loc.city_type, "locality")
        self.assertEqual(loc.country_code, "PK")

    def test_prefecture_level_city_still_resolves(self):
        loc = pi.build_location(TOKYO)
        self.assertEqual(loc.formatted_addr, "Tokyo, Japan")
        self.assertEqual(loc.city_type, "administrative_area_level_1")
        self.assertEqual(loc.country_code, "JP")

    def test_city_state_does_not_repeat_its_name(self):
        loc = pi.build_location(SINGAPORE)
        self.assertEqual(loc.formatted_addr, "Singapore")
        self.assertEqual(loc.city_type, "locality")
        self.assertEqual(loc.country_code, "SG")

    def test_country_and_continent_are_skipped(self):
        self.assertIsNone(pi.build_location(JAPAN))
        self.assertIsNone(pi.build_location(EUROPE))

    def test_garbage_is_skipped(self):
        self.assertIsNone(pi.build_location({}))
        self.assertIsNone(pi.build_location({"types": ["locality"], "address_components": []}))

    def test_payload_matches_contract(self):
        payload = pi.build_location(SAN_FRANCISCO).payload("en")
        self.assertEqual(payload, {
            "cityId": None,
            "formattedAddr": "San Francisco, CA, United States",
            "cityType": "locality",
            "countryCode": "US",
            "lang": "en",
        })


class PreloadFlowTests(unittest.IsolatedAsyncioTestCase):
    """End to end through a mocked transport: geocode → dedupe → POST."""

    def setUp(self):
        self.posts: list[httpx.Request] = []
        self.geocodes = 0

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.host == "maps.googleapis.com":
                self.geocodes += 1
                name = request.url.params.get("address", "").lower()
                result = GEOCODE_BY_NAME.get(name)
                if result is None:
                    return httpx.Response(200, json={"status": "ZERO_RESULTS", "results": []})
                return httpx.Response(200, json={"status": "OK", "results": [result]})
            self.posts.append(request)
            return httpx.Response(200, json={"ok": True})

        self._saved = (
            pi._client,
            settings.preload_inventory_enabled,
            settings.preload_inventory_base_url,
            settings.preload_inventory_callback_secret,
            settings.google_maps_api_key,
        )
        pi._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        settings.preload_inventory_enabled = True
        settings.preload_inventory_base_url = "https://backend.test/"
        settings.preload_inventory_callback_secret = "test-secret"
        settings.google_maps_api_key = "maps-key"
        pi._geocode_cache.clear()
        pi._recent_sends.clear()

    def tearDown(self):
        (pi._client, settings.preload_inventory_enabled, settings.preload_inventory_base_url,
         settings.preload_inventory_callback_secret, settings.google_maps_api_key) = self._saved
        pi._geocode_cache.clear()
        pi._recent_sends.clear()

    async def test_one_post_per_location_with_secret_header(self):
        sent = await pi.preload_locations(["San Diego", "Karachi", "San Francisco"], thread_id="t1")
        # "San Diego" is not in the fixture → ZERO_RESULTS → skipped
        self.assertEqual(sent, 2)
        self.assertEqual(len(self.posts), 2)
        for req in self.posts:
            self.assertEqual(str(req.url), "https://backend.test/trip-planner/preloadInventory")
            self.assertEqual(req.method, "POST")
            self.assertEqual(req.headers["X-Callback-Secret"], "test-secret")
        bodies = sorted(json.loads(r.content)["formattedAddr"] for r in self.posts)
        self.assertEqual(bodies, ["Karachi, Pakistan", "San Francisco, CA, United States"])

    async def test_country_level_destination_is_not_preloaded(self):
        sent = await pi.preload_locations(["Japan", "Europe"], thread_id="t1")
        self.assertEqual(sent, 0)
        self.assertEqual(self.posts, [])

    async def test_multi_city_explore_location_preloads_every_city(self):
        # The explore path hands over ONE string for "a trip to Barcelona and Madrid".
        sent = await pi.preload_locations(["Barcelona, Madrid"], thread_id="t1")
        self.assertEqual(sent, 2)
        bodies = sorted(json.loads(r.content)["formattedAddr"] for r in self.posts)
        self.assertEqual(bodies, ["Barcelona, Spain", "Madrid, Spain"])

    async def test_city_with_its_state_or_country_preloads_only_the_city(self):
        sent = await pi.preload_locations(["San Francisco, CA", "Paris, France"], thread_id="t1")
        self.assertEqual(sent, 2)
        bodies = sorted(json.loads(r.content)["formattedAddr"] for r in self.posts)
        self.assertEqual(bodies, ["Paris, France", "San Francisco, CA, United States"])

    async def test_planner_list_of_cities_preloads_each(self):
        # The planner now keeps every named city instead of merging to "Spain".
        sent = await pi.preload_locations(["Barcelona", "Madrid"], thread_id="t1")
        self.assertEqual(sent, 2)

    async def test_same_thread_same_city_sends_once_but_new_thread_sends_again(self):
        await pi.preload_locations(["Karachi"], thread_id="t1")
        await pi.preload_locations(["karachi", "Karachi "], thread_id="t1")
        self.assertEqual(len(self.posts), 1)
        await pi.preload_locations(["Karachi"], thread_id="t2")
        self.assertEqual(len(self.posts), 2)
        # geocode result was memoised — one lookup for four mentions
        self.assertEqual(self.geocodes, 1)

    async def test_disabled_flag_sends_nothing(self):
        settings.preload_inventory_enabled = False
        self.assertEqual(await pi.preload_locations(["Karachi"], thread_id="t1"), 0)
        self.assertIsNone(pi.preload_locations_background(["Karachi"], thread_id="t1"))
        self.assertEqual(self.posts, [])

    async def test_missing_secret_sends_nothing(self):
        settings.preload_inventory_callback_secret = None
        self.assertEqual(await pi.preload_locations(["Karachi"], thread_id="t1"), 0)
        self.assertEqual(self.posts, [])

    async def test_backend_failure_never_raises(self):
        def failing(request: httpx.Request) -> httpx.Response:
            if request.url.host == "maps.googleapis.com":
                return httpx.Response(200, json={"status": "OK", "results": [KARACHI]})
            raise httpx.ConnectError("boom")

        pi._client = httpx.AsyncClient(transport=httpx.MockTransport(failing))
        self.assertEqual(await pi.preload_locations(["Karachi"], thread_id="t1"), 0)

    async def test_background_entry_point_returns_immediately_and_completes(self):
        task = pi.preload_locations_background(["San Francisco"], thread_id="t1", source="plan")
        self.assertIsNotNone(task)
        self.assertEqual(self.posts, [])          # not awaited yet
        await task
        self.assertEqual(len(self.posts), 1)


if __name__ == "__main__":
    sys.exit(unittest.main(verbosity=2))
