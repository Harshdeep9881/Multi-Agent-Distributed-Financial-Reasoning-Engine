import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnalysisAgent:
    def __init__(self):
        self.portfolio = {'TSM': 0.12, '005930.KS': 0.10}  # Example: TSMC, Samsung

    def _currency_for_symbol(self, symbol):
        symbol = symbol.upper()
        if symbol.endswith("-USD"):
            return "USD"
        if symbol.endswith(".KS"):
            return "KRW"
        if symbol.endswith(".T"):
            return "JPY"
        if symbol.endswith(".HK"):
            return "HKD"
        if symbol.endswith(".NS"):
            return "INR"
        if symbol.endswith(".L"):
            return "GBP"
        if symbol.endswith(".DE") or symbol.endswith(".PA") or symbol.endswith(".AS") or symbol.endswith(".BR"):
            return "EUR"
        if symbol.endswith(".SW"):
            return "CHF"
        if symbol.endswith(".TO"):
            return "CAD"
        if symbol.endswith(".AX"):
            return "AUD"
        return "USD"

    def analyze_risk_exposure(self, market_data):
        """
        Analyze risk exposure based on portfolio and market data.
        
        Args:
            market_data: Dictionary with symbol keys and DataFrame values
            
        Returns:
            Dictionary with exposure data or empty dict if failed
        """
        try:
            total_aum = 1000000  # Example AUM
            exposure = {}
            
            logger.info(f"Analyzing risk exposure for portfolio: {list(self.portfolio.keys())}")
            logger.info(f"Market data available for: {list(market_data.keys())}")
            
            for symbol, weight in self.portfolio.items():
                try:
                    if symbol in market_data:
                        market_df = market_data[symbol]
                        
                        # Handle DataFrame or other data types
                        if isinstance(market_df, pd.DataFrame):
                            if market_df.empty:
                                logger.warning(f"Empty DataFrame for {symbol}")
                                price = 100.0  # Fallback price
                            else:
                                # Try to get the latest Close price
                                if 'Close' in market_df.columns:
                                    price = float(market_df['Close'].iloc[-1])
                                elif 'close' in market_df.columns:
                                    price = float(market_df['close'].iloc[-1])
                                else:
                                    # Use first numeric column as fallback
                                    numeric_cols = market_df.select_dtypes(include=['float64', 'int64']).columns
                                    if len(numeric_cols) > 0:
                                        price = float(market_df[numeric_cols[0]].iloc[-1])
                                    else:
                                        logger.warning(f"No numeric columns for {symbol}")
                                        price = 100.0
                        else:
                            logger.warning(f"Unexpected data type for {symbol}: {type(market_df)}")
                            price = 100.0
                        
                        exposure[symbol] = {
                            'weight': weight,
                            'value': weight * total_aum,
                            'price': price,
                            'currency': self._currency_for_symbol(symbol)
                        }
                        logger.info(f"Exposure for {symbol}: weight={weight:.2%}, price=${price:.2f}")
                    else:
                        logger.warning(f"Symbol {symbol} not in market data")
                        # Still add it with default price
                        exposure[symbol] = {
                            'weight': weight,
                            'value': weight * total_aum,
                            'price': 100.0,
                            'currency': self._currency_for_symbol(symbol)
                        }
                        
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {str(e)}")
                    exposure[symbol] = {
                        'weight': weight,
                        'value': weight * total_aum,
                        'price': 100.0,
                        'currency': self._currency_for_symbol(symbol)
                    }
            
            logger.info(f"Risk exposure analysis complete: {len(exposure)} positions")
            return exposure
            
        except Exception as e:
            logger.error(f"Error analyzing risk exposure: {str(e)}", exc_info=True)
            return {}
