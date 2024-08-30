SELECT COUNT(*)
FROM comments AS c,
  postHistory AS ph,
  votes AS v,
  users AS u
WHERE v.UserId = u.Id
  AND c.UserId = u.Id
  AND ph.UserId = u.Id
  AND c.Score = 0
  AND c.CreationDate >= CAST('2010-07-19 19:56:21' AS timestamp)
  AND c.CreationDate <= CAST('2014-09-11 13:36:12' AS timestamp)
  AND u.Views <= 433
  AND u.DownVotes >= 0
  AND u.CreationDate <= CAST('2014-09-12 21:37:39' AS timestamp);
