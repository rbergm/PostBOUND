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
  AND v.BountyAmount <= 300
  AND v.CreationDate >= CAST('2010-07-29 00:00:00' AS timestamp)
  AND u.UpVotes >= 0
  AND u.UpVotes <= 18;
