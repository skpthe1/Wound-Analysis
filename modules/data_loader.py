import pandas as pd
import pandera as pa
from pandera.typing import Series

class WoundDataSchema(pa.DataFrameModel):
    ID: Series[int] = pa.Field(ge=0)
    WEEK: Series[str] = pa.Field(str_matches=r"Week \d+")
    WOUND_TYPE: Series[str]
    NAME: Series[str]
    TOTAL_WOUND_AREA: Series[float] = pa.Field(ge=0)
    WOUND_COUNT: Series[int] = pa.Field(ge=0)
    AVG_WOUND_AREA: Series[float] = pa.Field(ge=0)
    ACTIVE_STATUS: Series[int] = pa.Field(isin=[0, 1])
    DW_CREATED_BY: Series[str]
    DW_UPDATED_BY: Series[str]
    DW_CREATION_TIMESTAMP: Series[pd.Timestamp]
    DW_UPDATED_TIMESTAMP: Series[pd.Timestamp]

    @pa.check("AVG_WOUND_AREA")
    def check_avg_wound_area(cls, series: Series[float]) -> Series[bool]:
        return series == series.groupby(level=0).transform("first")

    class Config:
        coerce = True

def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(
            file_path,
            parse_dates=['DW_CREATION_TIMESTAMP', 'DW_UPDATED_TIMESTAMP'],
            date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S.%f Z', utc=True)
        )
        
        # Convert WEEK to numeric
        df['WEEK_NUM'] = pd.to_numeric(
            df['WEEK'].str.extract('(\d+)')[0], 
            errors='coerce'
        ).dropna().astype(int)
        
        validated_df = WoundDataSchema.validate(df)
        
        # Create ordered categorical week
        validated_df['WEEK'] = pd.Categorical(
            validated_df['WEEK'],
            categories=sorted(validated_df['WEEK'].unique(), 
                            key=lambda x: int(x.split()[-1])),
            ordered=True
        )
        
        return validated_df

    except pa.errors.SchemaError as e:
        raise ValueError(f"Data validation failed: {str(e)}") from e
    except pd.errors.ParserError as e:
        raise ValueError(f"Date parsing error: {str(e)}") from e
    except Exception as e:
        raise ValueError(f"Unexpected error: {str(e)}") from e
