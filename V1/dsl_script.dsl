use teams.csv
model model1 = new SGDRegressor()
train.model1(teams, ["athletes", "prev_medals"], "medals")
predict.model1(teams)
error_ratio.model1(teams)
