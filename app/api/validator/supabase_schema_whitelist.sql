-- ============================================================
-- whitelist_domains table
-- Stores verified/trusted sources per category + country.
-- A source is "verified" when all 3 LLMs (OpenAI, Perplexity,
-- Gemini) independently name it as a top source.
-- ============================================================

create table public.whitelist_domains (
    id          uuid        default gen_random_uuid() primary key,
    category    text        not null,
    country     text        not null,
    source      text        not null,
    verified_at timestamptz default now() not null,
    created_at  timestamptz default now() not null,

    -- Enforce lowercase to prevent duplicate casing issues
    constraint lowercase_category check (category = lower(category)),
    constraint lowercase_country  check (country  = lower(country)),
    constraint lowercase_source   check (source   = lower(source)),

    -- One row per unique (category, country, source) triple
    constraint unique_category_country_source
        unique (category, country, source)
);

-- ── Indexes ────────────────────────────────────────────────────────────────

-- Primary lookup: "is this source trusted for category X in country Y?"
create index idx_whitelist_cat_country_source
    on public.whitelist_domains (category, country, source);

-- Lookup by country alone (e.g. load all trusted sources for "pakistan")
create index idx_whitelist_country
    on public.whitelist_domains (country);

-- Lookup by source alone (e.g. check if "foodpanda.com" is globally trusted)
create index idx_whitelist_source
    on public.whitelist_domains (source);

-- Lookup by category alone (e.g. all restaurant sources across countries)
create index idx_whitelist_category
    on public.whitelist_domains (category);

-- ── Row-level security (optional — enable if your project uses RLS) ────────
-- alter table public.whitelist_domains enable row level security;

-- ── Comments ───────────────────────────────────────────────────────────────
comment on table  public.whitelist_domains             is 'Verified trusted sources per travel category and country. Populated by the whitelist domain finder service.';
comment on column public.whitelist_domains.id          is 'Auto-generated UUID primary key.';
comment on column public.whitelist_domains.category    is 'Travel category in lowercase (e.g. restaurant, hotel, activity, place).';
comment on column public.whitelist_domains.country     is 'Country in lowercase (e.g. pakistan, france, japan). Always country-level, never city-level.';
comment on column public.whitelist_domains.source      is 'Verified domain link in lowercase (e.g. tripadvisor.com, yelp.com, foodpanda.com). Root domain only — no www., no paths, no protocol.';
comment on column public.whitelist_domains.verified_at is 'Timestamp when this source was last verified by the LLM consensus check.';
comment on column public.whitelist_domains.created_at  is 'Timestamp when this row was first inserted.';
