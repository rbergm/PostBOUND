SELECT COUNT(*)
FROM comments AS c,
  posts AS p,
  votes AS v,
  users AS u
WHERE u.Id = c.UserId
  AND u.Id = p.OwnerUserId
  AND u.Id = v.UserId
  AND c.Score = 0
  AND c.CreationDate <= CAST('2014-09-13 20:12:15' AS timestamp)
  AND p.CreationDate >= CAST('2010-07-27 01:51:15' AS timestamp)
  AND v.BountyAmount <= 50
  AND v.CreationDate <= CAST('2014-09-12 00:00:00' AS timestamp)
  AND u.UpVotes >= 0
  AND u.UpVotes <= 12
  AND u.CreationDate >= CAST('2010-07-19 19:09:39' AS timestamp);
