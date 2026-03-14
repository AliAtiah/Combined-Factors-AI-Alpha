"""
Data pipeline: bundle registration, pipeline engine, and pricing retrieval.
"""

import os
import pandas as pd
from zipline.data import bundles
from zipline.data.data_portal import DataPortal
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.loaders import USEquityPricingLoader
from zipline.pipeline.engine import SimplePipelineEngine
from zipline.utils.calendars import get_calendar

from config import EOD_BUNDLE_NAME


class PricingLoader:
    def __init__(self, bundle_data):
        self.loader = USEquityPricingLoader(
            bundle_data.equity_daily_bar_reader,
            bundle_data.adjustment_reader,
        )

    def get_loader(self, column):
        if column not in USEquityPricing.columns:
            raise ValueError(f"Column {column} not in USEquityPricing")
        return self.loader


def register_bundle(data_root=None):
    """Register the EOD data bundle with Zipline."""
    if data_root is None:
        data_root = os.path.join(os.getcwd(), "..", "..", "data", "project_7_eod")
    os.environ["ZIPLINE_ROOT"] = data_root

    ingest_func = bundles.csvdir.csvdir_equities(["daily"], EOD_BUNDLE_NAME)
    bundles.register(EOD_BUNDLE_NAME, ingest_func)


def build_pipeline_engine(bundle_data, trading_calendar):
    """Build and return a SimplePipelineEngine."""
    pricing_loader = PricingLoader(bundle_data)
    return SimplePipelineEngine(
        get_loader=pricing_loader.get_loader,
        calendar=trading_calendar.all_sessions,
        asset_finder=bundle_data.asset_finder,
    )


def get_pricing(data_portal, trading_calendar, assets, start_date, end_date, field="close"):
    """Retrieve historical pricing from the data portal."""
    end_dt = pd.Timestamp(end_date.strftime("%Y-%m-%d"), tz="UTC", offset="C")
    start_dt = pd.Timestamp(start_date.strftime("%Y-%m-%d"), tz="UTC", offset="C")

    end_loc = trading_calendar.closes.index.get_loc(end_dt)
    start_loc = trading_calendar.closes.index.get_loc(start_dt)

    return data_portal.get_history_window(
        assets=assets,
        end_dt=end_dt,
        bar_count=end_loc - start_loc,
        frequency="1d",
        field=field,
        data_frequency="daily",
    )


def load_bundle_and_engine():
    """Convenience function: load the bundle and return (bundle_data, engine, calendar, data_portal)."""
    trading_calendar = get_calendar("NYSE")
    bundle_data = bundles.load(EOD_BUNDLE_NAME)
    engine = build_pipeline_engine(bundle_data, trading_calendar)

    data_portal = DataPortal(
        bundle_data.asset_finder,
        trading_calendar=trading_calendar,
        first_trading_day=bundle_data.equity_daily_bar_reader.first_trading_day,
        equity_minute_reader=None,
        equity_daily_reader=bundle_data.equity_daily_bar_reader,
        adjustment_reader=bundle_data.adjustment_reader,
    )
    return bundle_data, engine, trading_calendar, data_portal
