SELECT COUNT(*)
FROM comments AS c, votes AS v, users AS u
WHERE u.Id = c.UserId
  AND u.Id = v.UserId
  AND c.CreationDate >= CAST('2010-08-10 17:55:45' AS timestamp)
  AND u.Reputation >= 1
  AND u.Reputation <= 691;
