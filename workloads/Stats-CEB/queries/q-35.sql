SELECT COUNT(*)
FROM comments AS c,
  votes AS v,
  badges AS b,
  users AS u
WHERE u.Id = c.UserId
  AND u.Id = v.UserId
  AND u.Id = b.UserId
  AND c.Score = 0
  AND v.BountyAmount >= 0
  AND v.CreationDate <= CAST('2014-09-11 00:00:00' AS timestamp)
  AND u.DownVotes <= 57
  AND u.CreationDate >= CAST('2010-08-26 09:01:31' AS timestamp)
  AND u.CreationDate <= CAST('2014-08-10 11:01:39' AS timestamp);
