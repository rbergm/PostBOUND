SELECT COUNT(*)
FROM comments AS c,
  votes AS v,
  badges AS b,
  users AS u
WHERE u.Id = b.UserId
  AND u.Id = c.UserId
  AND u.Id = v.UserId
  AND c.Score = 1
  AND c.CreationDate >= CAST('2010-08-27 14:12:07' AS timestamp)
  AND v.VoteTypeId = 5
  AND v.CreationDate >= CAST('2010-07-19 00:00:00' AS timestamp)
  AND v.CreationDate <= CAST('2014-09-13 00:00:00' AS timestamp)
  AND b.Date <= CAST('2014-08-19 10:32:13' AS timestamp)
  AND u.Reputation >= 1;
