SELECT COUNT(*)
FROM comments AS c,
  posts AS p,
  votes AS v,
  badges AS b,
  users AS u
WHERE u.Id = c.UserId
  AND c.UserId = p.OwnerUserId
  AND p.OwnerUserId = v.UserId
  AND v.UserId = b.UserId
  AND c.Score = 1
  AND p.Score >= -1
  AND p.Score <= 29
  AND p.CreationDate >= CAST('2010-07-19 20:40:36' AS timestamp)
  AND p.CreationDate <= CAST('2014-09-10 20:52:30' AS timestamp)
  AND v.BountyAmount <= 50
  AND b.Date <= CAST('2014-08-25 19:05:46' AS timestamp)
  AND u.DownVotes <= 11
  AND u.CreationDate >= CAST('2010-07-31 17:32:56' AS timestamp)
  AND u.CreationDate <= CAST('2014-09-07 16:06:26' AS timestamp);
