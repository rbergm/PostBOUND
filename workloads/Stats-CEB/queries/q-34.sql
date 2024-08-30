SELECT COUNT(*)
FROM comments AS c,
  votes AS v,
  badges AS b,
  users AS u
WHERE u.Id = c.UserId
  AND u.Id = v.UserId
  AND u.Id = b.UserId
  AND c.Score = 1
  AND c.CreationDate >= CAST('2010-07-20 23:17:28' AS timestamp)
  AND u.CreationDate >= CAST('2010-07-20 01:27:29' AS timestamp);
