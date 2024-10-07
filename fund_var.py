import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import argparse


class FundVaRComputer:
    def __init__(self, excel_data_file):
        self.excel_data_file = excel_data_file
        self.fund_info = {}
        self.liability_sensitivity = {}
        self.asset_allocation = {}
        self.sector_daily_returns = pd.DataFrame()
        self.risk_factor_daily_changes = pd.DataFrame()
        self.return_data = {}
        self.monthly_returns_data = {}
        self.funding_ratio_df = None
        self.var_1d = None
        self.var_1y_scale = None
        self.var_1y_bootstrap = None
        self.corr = None
        self.var_risk_factor_contribution = None
        self.var_asset_class_contribution = None



    def load_data(self):
        def find_table_start(df, keyword):
            """Find the row index where the table starts after the keyword."""
            for idx, row in df.iterrows():
                if keyword in str(row.values):
                    return idx + 1  # Return the next row after the keyword
            return -1  # Return -1 if keyword is not found
        xls = pd.ExcelFile(self.excel_data_file)
        # Load the 'Summary' sheet
        summary_df = pd.read_excel(xls, sheet_name='Summary')
        # Find the table start locations
        fund_info_start = find_table_start(summary_df, "Fund Information")
        liability_sensitivities_start = find_table_start(summary_df, "Liability Sensitivities")
        asset_allocation_start = find_table_start(summary_df, "Asset Allocation")
        ## extract fund information table
        fund_info_df = summary_df.iloc[fund_info_start:fund_info_start + 3, 1:4]
        self.fund_info["Q1"] = {"total_asset":fund_info_df.iloc[1, 1], "total_liability":fund_info_df.iloc[2, 1]}
        self.fund_info["Q2"] = {"total_asset": fund_info_df.iloc[1, 2], "total_liability": fund_info_df.iloc[2, 2]}
        ## extract liability senstivity table
        liability_sensitivities_df = summary_df.iloc[liability_sensitivities_start:liability_sensitivities_start + 3, 1:4]
        self.liability_sensitivity["Q1"] = {"CADIR30Y":liability_sensitivities_df.iloc[1,1], "CADInflation30Y":liability_sensitivities_df.iloc[2,1]}
        self.liability_sensitivity["Q2"] = {"CADIR30Y": liability_sensitivities_df.iloc[1, 2],
                                            "CADInflation30Y": liability_sensitivities_df.iloc[2, 2]}
        # Extract Asset Allocation
        asset_allocation_df = summary_df.iloc[asset_allocation_start:asset_allocation_start + 10, 1:6]
        cases = ["Actual","Benchmark"]
        for i in range(1,3):
            self.asset_allocation["Q{}".format(i)] = {}
            for k in range(0,2):
                self.asset_allocation["Q{}".format(i)][cases[k]] = {}
                for j in range(0,5):
                    row_index = j + 2
                    column_index = (i-1)*2+1+k
                    self.asset_allocation["Q{}".format(i)][cases[k]][asset_allocation_df.iloc[row_index,0]] = asset_allocation_df.iloc[row_index, column_index]

        ## load returns data
        for q_day in ["Q1","Q2"]:
            self.return_data[q_day] = {}
            data_df = pd.read_excel(xls, sheet_name='{} Data'.format(q_day), header=[0,1])
            data_df.columns = pd.MultiIndex.from_tuples(
                [(level1.strip(), level2.strip()) for level1, level2 in data_df.columns])
            data_df[("Unnamed:0_level_0", "Date")] = pd.to_datetime(data_df[('Unnamed: 0_level_0', 'Date')])
            data_df.set_index(('Unnamed: 0_level_0', 'Date'), inplace=True)
            name_map = {'Actual Investment Historical Daily Forward Looking Returns':"Actual",'Benchmark Historical Daily Forward Looking Returns':"Benchmark",'Market Risk Factor Historical Daily Changes':"RFChanges", "Market Risk Factor Daily Changes":"RFChanges"}
            for return_name in ['Actual Investment Historical Daily Forward Looking Returns','Benchmark Historical Daily Forward Looking Returns','Market Risk Factor Historical Daily Changes','Market Risk Factor Daily Changes']:
                try:
                    self.return_data[q_day][name_map[return_name]] = data_df.xs(return_name, level=0,axis=1)
                except:
                    continue

        ## build total asset daily returns based on asset allocation weights and asset class daily returns
        for i, return_data in self.return_data.items():
            for key in ["Actual", "Benchmark"]:
                weighted_columns = return_data[key] * return_data[key].columns.map(self.asset_allocation[i][key])
                return_data[key]["Total"] = weighted_columns.sum(axis=1)

    def compute_fund_ratio(self):
        res_list = []
        for i, asset_lib in self.fund_info.items():
            funding_ratio = asset_lib["total_asset"] / asset_lib["total_liability"]
            fund_surplus = asset_lib["total_asset"] - asset_lib["total_liability"]
            res_list.append({"Day":i, "Funding Ratio": funding_ratio, "Surplus of the fund": fund_surplus })
        self.funding_ratio_df = pd.DataFrame(res_list)

    def generate_month_returns(self):
        for i,return_data in self.return_data.items():
            self.monthly_returns_data[i] = {}
            for key in ["Actual", "Benchmark"]:
                # Calculate cumulative returns for each asset
                cumulative_returns_df = (1 + return_data[key]).cumprod()
                # Resample by month and get the last cumulative return for each month
                monthly_returns_df = cumulative_returns_df.resample('ME').last()
                # Calculate monthly returns for each asset
                self.monthly_returns_data[i][key] = monthly_returns_df.pct_change()
                ## handle the first month return
                self.monthly_returns_data[i][key].iloc[0] = (
                            monthly_returns_df.iloc[0] - 1)  # Compute the first month return

            ## generate 30y and 30 y inflation monthly change
            rate_month_changes_df = return_data["RFChanges"][["CAD IR 30yr","CAD Inflation 30yr"]].resample('ME').sum()
            #self.monthly_returns_data[i]["RFChanges"] = rate_month_changes_df.reset_index()
            self.monthly_returns_data[i]["RFChanges"] = rate_month_changes_df

    # Bootstrap function to simulate 1-year PnLs for both assets and liabilities
    def bootstrap_1year_pnls_df(self, df, n_simulations = 10000, n_days = 252):
        simulated_pnls_list = []
        np.random.seed(7)
        for _ in range(n_simulations):
            # Sample 252 rows (both Asset_PnL and Liability_PnL for each day) with replacement
            sampled_pnls = df.sample(n=n_days, replace=True)
            # Calculate cumulative 1-year PnLs by summing the daily PnLs for assets and liabilities
            cumulative_asset_pnl = sampled_pnls['TotalAsset'].sum()
            cumulative_liability_pnl = sampled_pnls['TotalLibility'].sum()
            cumulative_bm_pnl = sampled_pnls['TotalBenchmark'].sum()
            # Store the cumulative PnLs for each simulation
            simulated_pnls_list.append({
                'TotalAsset': cumulative_asset_pnl,
                'TotalLibility': cumulative_liability_pnl,
                'TotalBenchmark':cumulative_bm_pnl
            })
        # Return the results as a DataFrame
        return pd.DataFrame(simulated_pnls_list)

    def calculate_var(self, returns, confidence_level=0.95):
        """Calculate the VaR for given returns."""
        return np.percentile(returns.dropna(), (1 - confidence_level) * 100)

    def compute_1day_var(self):
        res_list = []
        for i, return_data in self.return_data.items():
            #self.var_1d[i] = {}
            temp_dict = {"Date":i}
            ## compute total asset daily pnls
            actual_asset_pnls_daily= return_data["Actual"]["Total"] * self.fund_info[i]["total_asset"]
            bm_asset_pnls_daily = return_data["Benchmark"]["Total"] * self.fund_info[i]["total_asset"]
            active_pnls_daily = actual_asset_pnls_daily - bm_asset_pnls_daily
            ## compute 1-day total asset var
            temp_dict["TotalAsset"] = -self.calculate_var(actual_asset_pnls_daily)
            temp_dict["TotalBenchmark"] = -self.calculate_var(bm_asset_pnls_daily)
            temp_dict["TotalActive"] = -self.calculate_var(active_pnls_daily)
            ## compute total libility var
            ## approximate libility daily pnl using sensitiviy
            liability_pnls_daily = ( return_data["RFChanges"]["CAD IR 30yr"] * 10000 * self.liability_sensitivity[i]["CADIR30Y"] + return_data["RFChanges"]["CAD Inflation 30yr"] * 10000 * self.liability_sensitivity[i]["CADInflation30Y"] ) / 1000
            temp_dict["TotalLibility"] = -self.calculate_var(-liability_pnls_daily)


            ## compute funding ratio and surplus var
            funding_ratio_daily = (actual_asset_pnls_daily + self.fund_info[i]["total_asset"]) / ( liability_pnls_daily + self.fund_info[i]["total_liability"])
            temp_dict["FundingRatio"] = self.calculate_var(funding_ratio_daily)
            fund_surplus_daily = (actual_asset_pnls_daily + self.fund_info[i]["total_asset"]) - ( liability_pnls_daily + self.fund_info[i]["total_liability"])
            temp_dict["FundSurplus"] = self.calculate_var(fund_surplus_daily)
            #self.var_1d[i]["FundingRatio"] = (self.fund_info[i]["total_asset"] - self.var_1d[i]["TotalAsset"] ) /  (self.fund_info[i]["total_liability"] + self.var_1d[i]["TotalLibility"] )
            #self.var_1d[i]["FundSurplus"] = (self.fund_info[i]["total_asset"] - self.var_1d[i]["TotalAsset"]) - ( self.fund_info[i]["total_liability"] + self.var_1d[i]["TotalLibility"])
            res_list.append(temp_dict)
        self.var_1d = pd.DataFrame(res_list)[["Date","TotalAsset","TotalLibility","FundingRatio","FundSurplus","TotalBenchmark","TotalActive"]]

    def compute_1year_var_scale(self):
        ## generate monthly returns based on daily returns
        self.generate_month_returns()
        res_list = []
        for i, return_data in self.monthly_returns_data.items():
            temp_dict = {"Date":i}
            ## compute total asset monthly pnls
            actual_asset_pnls_monthly =  return_data["Actual"]["Total"] * self.fund_info[i]["total_asset"]
            bm_asset_pnls_monthly = return_data["Benchmark"]["Total"] * self.fund_info[i]["total_asset"]
            active_asset_pnls_monthly = actual_asset_pnls_monthly - bm_asset_pnls_monthly
            # compute the 1month var total asset
            # scale the 1month var by sqrt(12) to generate 1 year var
            temp_dict["TotalAsset"] = -self.calculate_var(actual_asset_pnls_monthly) * np.sqrt(12)
            temp_dict["TotalBenchmark"] = -self.calculate_var(bm_asset_pnls_monthly) * np.sqrt(12)
            temp_dict["TotalActive"] = -self.calculate_var(active_asset_pnls_monthly) * np.sqrt(12)

            ## compute total libility var
            liability_pnls_monthly = (return_data["RFChanges"]["CAD IR 30yr"] * 10000 * self.liability_sensitivity[i]["CADIR30Y"] + return_data["RFChanges"]["CAD Inflation 30yr"] * 10000 * self.liability_sensitivity[i]["CADInflation30Y"])/1000
            liability_var_1m = -self.calculate_var(-liability_pnls_monthly)
            temp_dict["TotalLibility"] = liability_var_1m * np.sqrt(12)
            ## compute funding ratio and surplus var
            funding_ratio_monthly = (actual_asset_pnls_monthly * np.sqrt(12) + self.fund_info[i]["total_asset"]) / ( liability_pnls_monthly* np.sqrt(12) + self.fund_info[i]["total_liability"])
            temp_dict["FundingRatio"] = self.calculate_var(funding_ratio_monthly)
            #print(funding_ratio_monthly)
            fund_surplus_monthly = (actual_asset_pnls_monthly* np.sqrt(12) + self.fund_info[i]["total_asset"]) - ( liability_pnls_monthly* np.sqrt(12) + self.fund_info[i]["total_liability"])
            temp_dict["FundSurplus"] = self.calculate_var(fund_surplus_monthly)

            res_list.append(temp_dict)
        self.var_1y_scale = pd.DataFrame(res_list)[["Date","TotalAsset","TotalLibility","FundingRatio","FundSurplus","TotalBenchmark","TotalActive"]]

    def compute_1y_var_bootstrap(self):
        res_list = []
        for i, return_data in self.return_data.items():
            # self.var_1d[i] = {}
            temp_dict = {"Date": i}
            ## compute total asset daily pnls
            actual_asset_pnls_daily = return_data["Actual"]["Total"] * self.fund_info[i]["total_asset"]
            bm_asset_pnls_daily = return_data["Benchmark"]["Total"] * self.fund_info[i]["total_asset"]
            liability_pnls_daily = (return_data["RFChanges"]["CAD IR 30yr"] * 10000 * self.liability_sensitivity[i][
                "CADIR30Y"] + return_data["RFChanges"]["CAD Inflation 30yr"] * 10000 * self.liability_sensitivity[i][
                                        "CADInflation30Y"]) / 1000

            daily_pnls_df = pd.concat([actual_asset_pnls_daily, bm_asset_pnls_daily, liability_pnls_daily], axis=1)
            daily_pnls_df.columns = ["TotalAsset", "TotalBenchmark", "TotalLibility"]
            ## bootstrap 1year pnls
            yearly_pnls_df = self.bootstrap_1year_pnls_df(daily_pnls_df)
            yearly_pnls_df["TotalActive"] = yearly_pnls_df["TotalAsset"] - yearly_pnls_df["TotalBenchmark"]
            yearly_pnls_df["FundingRatio"] = (self.fund_info[i]["total_asset"] + yearly_pnls_df["TotalAsset"]) / (self.fund_info[i]["total_liability"] + yearly_pnls_df["TotalLibility"])
            yearly_pnls_df["FundSurplus"] = (self.fund_info[i]["total_asset"] + yearly_pnls_df["TotalAsset"]) - (self.fund_info[i]["total_liability"] + yearly_pnls_df["TotalLibility"])
            temp_dict["TotalAsset"] = -self.calculate_var(yearly_pnls_df["TotalAsset"])
            temp_dict["TotalBenchmark"] = -self.calculate_var(yearly_pnls_df["TotalBenchmark"])
            temp_dict["TotalActive"] = -self.calculate_var(yearly_pnls_df["TotalActive"])
            temp_dict["TotalLibility"] = -self.calculate_var(-yearly_pnls_df["TotalLibility"])
            temp_dict["FundingRatio"] = self.calculate_var(yearly_pnls_df["FundingRatio"])
            temp_dict["FundSurplus"] = self.calculate_var(yearly_pnls_df["FundSurplus"])
            res_list.append(temp_dict)
        self.var_1y_bootstrap = pd.DataFrame(res_list)[["Date", "TotalAsset", "TotalLibility", "FundingRatio", "FundSurplus", "TotalBenchmark", "TotalActive"]]

    def break_down_var_asset_class(self):
        res_list = []
        for i, return_data in self.return_data.items():
            asset_contribution = {"Date":i}
            # Calculate the standard deviation (volatility) of each asset class
            volatility = return_data["Actual"][['Fixed Income', 'Real Estate', 'Public Equity', 'Private Equity','Alpha Strategy']].std()
            # Calculate the risk contribution for each asset class
            risk_contribution = pd.Series(self.asset_allocation[i]["Actual"]) * volatility  # Exclude the intercept
            # Normalize the risk contributions (proportional contribution to total VaR)
            total_risk_contribution = risk_contribution.sum()
            risk_contribution_percent = risk_contribution / total_risk_contribution
            risk_contribution_percent["Date"] = i
            res_list.append(risk_contribution_percent)
        self.var_asset_class_contribution = pd.DataFrame(res_list)


    def break_down_var_risk_factor(self, risk_factors = ['SPX', 'FX USD/CAD', 'CAD IR 2yr', 'CAD IR 10yr', 'CAD IR 30yr', 'USD IR 2yr', 'USD IR 10yr', 'USD IR 30yr', 'CAD Inflation 30yr','USD Inflation 30yr', 'USD IG 5Y Spread', 'USD HY 5Y Spread', 'VIX']):
        res_list = []
        for i, return_data in self.return_data.items():
            rf_contributions = {"Date":i}
            risk_factor_returns = return_data["RFChanges"][risk_factors]
            x = risk_factor_returns
            y = return_data["Actual"]["Total"]
            x = sm.add_constant(x)
            # Perform linear regression
            model = sm.OLS(y, x).fit()
            # Get the regression coefficients (the sensitivities of the portfolio to each asset class)
            coefficients = model.params
            # Calculate the standard deviation (volatility) of each asset class
            volatility = risk_factor_returns.std()
            # Calculate the risk contribution for each asset class
            risk_contribution = np.abs(coefficients[1:]) * volatility
            # Normalize the risk contributions (proportional contribution to total VaR)
            total_risk_contribution = risk_contribution.sum()
            risk_contribution_percent = risk_contribution / total_risk_contribution
            risk_contribution_percent["Date"] = i
            res_list.append(risk_contribution_percent)
        self.var_risk_factor_contribution = pd.DataFrame(res_list)



    def compute_corr(self):
        self.corr = {}
        for i, return_data in self.return_data.items():
            asset_return_df = return_data["Actual"][['Fixed Income', 'Real Estate', 'Public Equity', 'Private Equity','Alpha Strategy']]

            liability_pnls_daily = (return_data["RFChanges"]["CAD IR 30yr"] * 10000 * self.liability_sensitivity[i][
                "CADIR30Y"] + return_data["RFChanges"]["CAD Inflation 30yr"] * 10000 *
                                        self.liability_sensitivity[i][
                                            "CADInflation30Y"]) / 1000
            return_df = pd.concat([asset_return_df, liability_pnls_daily], axis=1)
            return_df.columns = ['Fixed Income', 'Real Estate', 'Public Equity', 'Private Equity','Alpha Strategy', 'Liability']
            self.corr[i] = return_df.corr()


    def write_result(self, out_dir = None):
        out_file = "VaR_Out.xlsx"
        if out_dir is not None:
            out_file = os.path.join(out_dir, out_file)
        with pd.ExcelWriter(out_file) as writer:
            self.funding_ratio_df.to_excel(writer, sheet_name='FundingRatio_Surpus', index=False)
            self.var_1d.to_excel(writer, sheet_name='VaR1D', index=False)
            self.var_1y_scale.to_excel(writer, sheet_name='VaR1Y_Scale', index=False)
            self.var_1y_bootstrap.to_excel(writer, sheet_name='VaR1Y_Bootstrap', index=False)
            self.var_asset_class_contribution.to_excel(writer, sheet_name='VaR_Asset_Breakdown', index=False)
            self.var_risk_factor_contribution.to_excel(writer, sheet_name='VaR_RF_Breakdown', index=False)
            self.corr["Q1"].to_excel(writer, sheet_name='Corr_Q1', index=True)
            self.corr["Q2"].to_excel(writer, sheet_name='Corr_Q2', index=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filepath', type=str, help='file path for the excel data spreadsheet', default="Market Risk - Mini Project.xlsx")
    parser.add_argument('-o', '--outdir', type=str, help='out dir path', default=None)
    args = parser.parse_args()

    file_path = args.filepath
    out_dir = args.outdir
    fund_var_computer = FundVaRComputer(file_path)
    fund_var_computer.load_data()
    fund_var_computer.compute_fund_ratio()
    fund_var_computer.compute_1day_var()
    fund_var_computer.compute_1year_var_scale()
    fund_var_computer.compute_1y_var_bootstrap()
    fund_var_computer.break_down_var_asset_class()
    fund_var_computer.break_down_var_risk_factor()
    fund_var_computer.compute_corr()
    fund_var_computer.write_result()
