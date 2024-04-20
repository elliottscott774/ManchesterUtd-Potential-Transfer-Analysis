Abstract or Overview:
- The purpose of the app was to create a transfer recommendation system for football clubs.
  Through the app, a club can select the areas of performance they want to improve via a transfer,
  and using a decision tree model, the app recommends several players that fit the needs of the club.
  Football fans and potentially football clubs might find the app useful for creating short lists of players to futher scout,
  as well as understand the profile of player that your favorite team is lacking.

Data Description:
- The data used for training in this project comes from: https://www.kaggle.com/datasets/davidcariboo/player-scores.
- Multiple functions were necessary to format the data. Team performance and player data were extracted and formed into squads, which is a team given a specific season.


Algorithm Description:
- To train the decision tree model, a dataset of teams needed to constructed. Each team represents a club and a season that the team competed in.
- Each team has some performance level, such as wins, loss, clean sheets, ect, which can be used as a target variable for a model.
- The decision tree model is trained by taking a team of player data as input and predicts the value of some performance metric.
- Using the trained model, a recommendation system can be built by transfering players into some squad of players and testing if the performance metric improved.
- The model returns the top 5 players which improved the performance metric the most.
![Transfer_Recommender](https://github.com/elliottscott774/ManchesterUtd-Potential-Transfer-Analysis/assets/123982857/4e89ee50-4903-40ea-9807-61c6921e8eb5)


Tools Used: 
- Streamlit for hosting app
- BackBlaze for hosting data
- Boto3
- Pandas
- Numpy
- Sklearn

Ethical Concerns: 
- A football team and a football transfer involves real humans that should not be simplified down via an algorithm.
  This model is a simplification of what is really happening, and should be treated as such.
- The data used for training the model comes from seasons 2012 - early 2023. Using the model far beyond in time to this period will result
   in recommendations with little sense.
- Some combinations of team and season will result in the model failing. This is because the model mandates a certain number of players in each position and these teams fail to meet this condition. 



