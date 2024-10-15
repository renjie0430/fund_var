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
        self.yearly_returns_data_bootstrapped = {}
        self.yearly_returns_data_bootstrapped_statistics = {}
        self.daily_return_data = {}
        self.daily_return_statistics = {}
        self.monthly_returns_data = {}
        self.funding_ratio_df = None
        self.var_1d = None
        self.returns_var_1d = None
        self.var_1y_scale = None
        self.var_1y_bootstrap = None
        self.returns_var_1y_bootstrap = None
        self.rf_corr = None
        self.asset_corr = None
        self.var_risk_factor_contribution = None
        self.var_asset_class_contribution = None
        self.asset_standard_alone_var = None



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
            data_df = pd.read_excel(xls, sheet_name='{} Data'.format(q_day), header=[0,1])
            data_df.columns = pd.MultiIndex.from_tuples(
                [(level1.strip(), level2.strip()) for level1, level2 in data_df.columns])
            data_df[("Unnamed:0_level_0", "Date")] = pd.to_datetime(data_df[('Unnamed: 0_level_0', 'Date')])
            data_df.set_index(('Unnamed: 0_level_0', 'Date'), inplace=True)


            name_map = {'Actual Investment Historical Daily Forward Looking Returns':"Actual",'Benchmark Historical Daily Forward Looking Returns':"Benchmark",'Market Risk Factor Historical Daily Changes':"RFChanges", "Market Risk Factor Daily Changes":"RFChanges"}
            data_df.columns = pd.MultiIndex.from_arrays(
                [data_df.columns.get_level_values(0).map(name_map),  # Apply the mapping
                 data_df.columns.get_level_values(1)]  # Keep the second level unchanged
            )
            data_df = data_df.loc[:, data_df.columns.get_level_values(0).isin(["Actual","Benchmark","RFChanges"])]
            # compute the active returns Actual - BM returns
            active_returns_df = data_df["Actual"] - data_df["Benchmark"]
            active_returns_df.columns = pd.MultiIndex.from_product([["Active"], data_df["Actual"].columns])
            data_df = pd.concat([data_df, active_returns_df], axis=1)

            self.daily_return_data[q_day] = data_df
            ## generate common stats for daily returns
            self.daily_return_statistics[q_day] = data_df.describe()

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

    # Bootstrap function to simulate 1-year returns based on 1 day returns
    ## for asset we assume the returns are log returns, for intrest rate and inflation we assume they are absolute movement
    def bootstrap_1year_returns_df(self, n_simulations = 10000, n_days = 252):
        np.random.seed(7)
        for i, daily_return_data in self.daily_return_data.items():
            bootstrap_results = []
            # Perform bootstrap sampling 100 times
            for j in range(n_simulations):
                # Sample with replacement
                bootstrap_sample = daily_return_data.sample(n=n_days, replace=False)
                # Sum the sampled rows and append the result
                bootstrap_sum = bootstrap_sample.sum(axis=0)
                bootstrap_results.append(bootstrap_sum)
            # Create a new DataFrame to store the results of all bootstrap iterations
            self.yearly_returns_data_bootstrapped[i] = pd.DataFrame(bootstrap_results, columns=daily_return_data.columns)
            self.yearly_returns_data_bootstrapped_statistics[i] = self.yearly_returns_data_bootstrapped[i].describe()

    def calculate_var(self, returns, confidence_level=0.95):
        """Calculate the VaR for given returns."""
        return np.percentile(returns.dropna(), (1 - confidence_level) * 100)


    def compute_returns_var(self, return_data_dict):
        res_list = []
        for i, return_data in return_data_dict.items():
            #self.var_1d[i] = {}
            temp_dict = {"Date":i}
            ## compute total asset daily pnls based on total asset and fund weights in differents assets
            ## here we asssume the returns are log returns which should be very close to percentage returns for 1 day
            actual_asset_returns =   np.exp( (return_data.xs('Actual', axis=1, level=0) * return_data.xs('Actual', axis=1, level=0).columns.map(self.asset_allocation[i]['Actual']) ).sum(axis=1) ) - 1
            bm_asset_returns =  np.exp( (return_data.xs('Benchmark', axis=1, level=0) * return_data.xs('Benchmark', axis=1, level=0).columns.map(self.asset_allocation[i]['Benchmark']) ).sum(axis=1) ) - 1
            active_returns = actual_asset_returns - bm_asset_returns
            ## compute 1-day total asset var
            temp_dict["TotalAssetReturn"] = -self.calculate_var(actual_asset_returns)
            temp_dict["TotalBenchmarkReturn"] = -self.calculate_var(bm_asset_returns)
            temp_dict["TotalActiveReturn"] = -self.calculate_var(active_returns)
            temp_dict["TotalAssetValue"] = self.fund_info[i]["total_asset"]
            ## compute funding ratio and surplus var here we need to take into account the correaltion between asset pnls and liablity pnls, so we cannot directly use asset var and liability var
            ## we need to sample asset and liability simultaneously
            res_list.append(temp_dict)
        res_df = pd.DataFrame(res_list)[["Date","TotalAssetValue","TotalAssetReturn","TotalBenchmarkReturn","TotalActiveReturn"]]
        return res_df

    def compute_pnl_var_from_returns(self, return_data_dict):
        res_list = []
        for i, return_data in return_data_dict.items():
            #self.var_1d[i] = {}
            temp_dict = {"Date":i}
            ## compute total asset daily pnls based on total asset and fund weights in differents assets
            ## here we asssume the returns are log returns which should be very close to percentage returns for 1 day
            actual_asset_pnls_daily= self.fund_info[i]["total_asset"] * ( np.exp( (return_data.xs('Actual', axis=1, level=0) * return_data.xs('Actual', axis=1, level=0).columns.map(self.asset_allocation[i]['Actual']) ).sum(axis=1) ) - 1)
            bm_asset_pnls_daily = self.fund_info[i]["total_asset"] * ( np.exp( (return_data.xs('Benchmark', axis=1, level=0) * return_data.xs('Benchmark', axis=1, level=0).columns.map(self.asset_allocation[i]['Benchmark']) ).sum(axis=1) ) - 1)
            active_pnls_daily = actual_asset_pnls_daily - bm_asset_pnls_daily
            ## compute 1-day total asset var
            temp_dict["TotalAsset"] = -self.calculate_var(actual_asset_pnls_daily)
            temp_dict["TotalBenchmark"] = -self.calculate_var(bm_asset_pnls_daily)
            temp_dict["TotalActive"] = -self.calculate_var(active_pnls_daily)
            ## compute total libility var
            ## approximate libility daily pnl using sensitiviy
            liability_pnls_daily = ( return_data.xs('RFChanges', axis=1, level=0)["CAD IR 30yr"] * 10000 * self.liability_sensitivity[i]["CADIR30Y"] + return_data.xs('RFChanges', axis=1, level=0)["CAD Inflation 30yr"] * 10000 * self.liability_sensitivity[i]["CADInflation30Y"] ) / 1000
            temp_dict["TotalLibility"] = -self.calculate_var(-liability_pnls_daily)
            ## compute funding ratio and surplus var here we need to take into account the correaltion between asset pnls and liablity pnls, so we cannot directly use asset var and liability var
            ## we need to sample asset and liability simultaneously
            funding_ratio_daily = (actual_asset_pnls_daily + self.fund_info[i]["total_asset"]) / ( liability_pnls_daily + self.fund_info[i]["total_liability"])
            temp_dict["FundingRatio"] = self.calculate_var(funding_ratio_daily)
            fund_surplus_daily = (actual_asset_pnls_daily + self.fund_info[i]["total_asset"]) - ( liability_pnls_daily + self.fund_info[i]["total_liability"])
            temp_dict["FundSurplus"] = self.calculate_var(fund_surplus_daily)
            #self.var_1d[i]["FundingRatio"] = (self.fund_info[i]["total_asset"] - self.var_1d[i]["TotalAsset"] ) /  (self.fund_info[i]["total_liability"] + self.var_1d[i]["TotalLibility"] )
            #self.var_1d[i]["FundSurplus"] = (self.fund_info[i]["total_asset"] - self.var_1d[i]["TotalAsset"]) - ( self.fund_info[i]["total_liability"] + self.var_1d[i]["TotalLibility"])
            res_list.append(temp_dict)
        res_df = pd.DataFrame(res_list)[["Date","TotalAsset","TotalLibility","FundingRatio","FundSurplus","TotalBenchmark","TotalActive"]]
        return res_df

    def compute_1day_var(self):
        res_df  = self.compute_pnl_var_from_returns(self.daily_return_data)
        res_df["VaRType"] = "VaR_1D"
        self.var_1d = res_df
        res_df  = self.compute_returns_var(self.daily_return_data)
        res_df["VaRType"] = "VaR_1D"
        self.returns_var_1d = res_df


    def compute_1year_var_bootstrap(self,n_simulations = 5000, n_days = 252):
        self.bootstrap_1year_returns_df(n_simulations, n_days)
        res_df = self.compute_pnl_var_from_returns(self.yearly_returns_data_bootstrapped)
        res_df["VaRType"] = "VaR_1Y_Bootstrap"
        self.var_1y_bootstrap = res_df
        res_df = self.compute_returns_var(self.yearly_returns_data_bootstrapped)
        res_df["VaRType"] = "VaR_1Y_Bootstrap"
        self.returns_var_1y_bootstrap = res_df


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
            ## it is not right to scale the pnl directly so i do not think we should use scaling rule for funding ratio and fund surplus var
            #funding_ratio_monthly = (actual_asset_pnls_monthly * np.sqrt(12) + self.fund_info[i]["total_asset"]) / ( liability_pnls_monthly* np.sqrt(12) + self.fund_info[i]["total_liability"])
            #temp_dict["FundingRatio"] = self.calculate_var(funding_ratio_monthly)
            #print(funding_ratio_monthly)
            #fund_surplus_monthly = (actual_asset_pnls_monthly* np.sqrt(12) + self.fund_info[i]["total_asset"]) - ( liability_pnls_monthly* np.sqrt(12) + self.fund_info[i]["total_liability"])
            #temp_dict["FundSurplus"] = self.calculate_var(fund_surplus_monthly)
            res_list.append(temp_dict)
        #self.var_1y_scale = pd.DataFrame(res_list)[["Date","TotalAsset","TotalLibility","FundingRatio","FundSurplus","TotalBenchmark","TotalActive"]]
        self.var_1y_scale = pd.DataFrame(res_list)[["Date","TotalAsset","TotalLibility","TotalBenchmark","TotalActive"]]

    def attribute_asset_var_to_asset_class(self):
        ## the var contribution from an individual asset class i can be approximated as
        ## contribution_i = w_i * \frac{Cov(R_i, R_p)}{Var(R_p)} * VaR_p
        res_list = []
        for i, return_data in self.daily_return_data.items():
            asset_contribution = {}
            total_returns =   np.exp( (return_data.xs('Actual', axis=1, level=0) * return_data.xs('Actual', axis=1, level=0).columns.map(self.asset_allocation[i]['Actual']) ).sum(axis=1) ) - 1
            total_returns_variance = np.var(total_returns)
            for asset_class in ['Fixed Income', 'Real Estate', 'Public Equity', 'Private Equity','Alpha Strategy']:
                cov = np.cov(return_data["Actual"][asset_class], total_returns)[0,1]
                asset_contribution[asset_class] = cov / total_returns_variance * self.asset_allocation[i]["Actual"][asset_class]
            asset_contribution["Date"] = i
            res_list.append(asset_contribution)
        self.var_asset_class_contribution = pd.DataFrame(res_list)

    def generate_asset_standard_alone_var(self):
        ## only keep one asset shocks but keep all other assets constant
        res_list = []
        for i, return_data in self.daily_return_data.items():
            var_contribution = {}
            for asset_class in ['Fixed Income', 'Real Estate', 'Public Equity', 'Private Equity','Alpha Strategy']:
                asset_returns = return_data.xs('Actual', axis=1, level=0)[asset_class]
                asset_var = - self.fund_info[i]["total_asset"] *  self.asset_allocation[i]['Actual'][asset_class] * self.calculate_var(asset_returns)
                var_contribution[asset_class] = asset_var
            var_contribution["Date"] = i
            res_list.append(var_contribution)
        self.asset_standard_alone_var = pd.DataFrame(res_list)

    def attribute_asset_var_to_risk_factor(self, risk_factors = ['SPX', 'FX USD/CAD', 'CAD IR 2yr', 'CAD IR 10yr', 'CAD IR 30yr', 'CAD Inflation 30yr', 'USD IG 5Y Spread', 'USD HY 5Y Spread', 'VIX']):
        res_list = []
        for i, return_data in self.daily_return_data.items():
            rf_contribution = {}
            total_returns = np.exp((return_data.xs('Actual', axis=1, level=0) * return_data.xs('Actual', axis=1, level=0).columns.map(self.asset_allocation[i]['Actual'])).sum(axis=1)) - 1
            total_returns_variance = np.var(total_returns)
            x = return_data["RFChanges"][risk_factors]
            y = total_returns
            x = sm.add_constant(x)
            # Perform linear regression
            model = sm.OLS(y, x).fit()
            # Get the regression coefficients (the sensitivities of the portfolio to each asset class)
            coefficients = model.params
            for j in range(len(risk_factors)):
                rf = risk_factors[j]
                rf_sensitivity = coefficients[rf]
                risk_factor_returns = return_data["RFChanges"][rf]
                cov = np.cov(total_returns, risk_factor_returns)[0,1]
                rf_contribution[rf] = cov / total_returns_variance * rf_sensitivity
            rf_contribution["Date"] = i
            res_list.append(rf_contribution)
        self.var_risk_factor_contribution = pd.DataFrame(res_list)


    def compute_rf_corr(self):
        self.rf_corr = {}
        for i, return_data in self.daily_return_data.items():
            rf_daily = return_data["RFChanges"]
            self.rf_corr[i] = rf_daily.corr()

    def compute_asset_liability_corr(self):
        self.asset_corr = {}
        for i, return_data in self.daily_return_data.items():
            asset_return_df = return_data["Actual"][['Fixed Income', 'Real Estate', 'Public Equity', 'Private Equity','Alpha Strategy']]

            liability_pnls_daily = (return_data["RFChanges"]["CAD IR 30yr"] * 10000 * self.liability_sensitivity[i][
                "CADIR30Y"] + return_data["RFChanges"]["CAD Inflation 30yr"] * 10000 *
                                        self.liability_sensitivity[i][
                                            "CADInflation30Y"]) / 1000
            liability_changes_daily = liability_pnls_daily / self.fund_info[i]['total_liability']
            return_df = pd.concat([asset_return_df, liability_changes_daily], axis=1)
            return_df.columns = ['Fixed Income', 'Real Estate', 'Public Equity', 'Private Equity','Alpha Strategy', 'Liability']
            self.asset_corr[i] = return_df.corr()


    def write_result(self, out_dir = None):
        out_file = "VaR_Out.xlsx"
        if out_dir is not None:
            out_file = os.path.join(out_dir, out_file)
        with pd.ExcelWriter(out_file) as writer:
            self.funding_ratio_df.to_excel(writer, sheet_name='FundingRatio_Surpus', index=False)
            self.var_1d.to_excel(writer, sheet_name='VaR', index=False, startrow=0)
            self.var_1y_bootstrap.to_excel(writer, sheet_name='VaR', index=False, startrow=4)
            self.returns_var_1d.to_excel(writer, sheet_name='VaR', index=False, startrow=8)
            self.returns_var_1y_bootstrap.to_excel(writer, sheet_name='VaR', index=False, startrow=12)
            self.var_asset_class_contribution.to_excel(writer, sheet_name='var_asset_attribution',index=False)
            self.var_risk_factor_contribution.to_excel(writer, sheet_name='var_rf_attribution',index=False)
            self.asset_standard_alone_var.to_excel(writer, sheet_name='var_1D_asset_stand_alone',index=False)
            for q_day in ["Q1","Q2"]:
                self.daily_return_statistics[q_day].to_excel(writer, sheet_name='return_stats_daily_{}'.format(q_day), startrow=0)

                self.yearly_returns_data_bootstrapped_statistics[q_day].to_excel(writer, sheet_name='return_stats_yearly_{}'.format(q_day),
                                                             startrow=0)

                self.asset_corr[q_day].to_excel(writer, sheet_name='corr_asset_liability_{}'.format(q_day), startrow=0)
                self.asset_corr[q_day].to_excel(writer, sheet_name='corr_asset_liability_{}'.format(q_day), startrow=0)
                self.rf_corr[q_day].to_excel(writer, sheet_name='corr_rf_{}'.format(q_day))
                self.rf_corr[q_day].to_excel(writer, sheet_name='corr_rf_{}'.format(q_day))


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
    fund_var_computer.compute_1year_var_bootstrap()
    fund_var_computer.attribute_asset_var_to_asset_class()
    fund_var_computer.attribute_asset_var_to_risk_factor()
    fund_var_computer.compute_rf_corr()
    fund_var_computer.compute_asset_liability_corr()
    fund_var_computer.generate_asset_standard_alone_var()
    fund_var_computer.write_result()
