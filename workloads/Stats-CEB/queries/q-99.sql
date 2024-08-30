SELECT COUNT(*)
FROM comments AS c,
  posts AS p,
  votes AS v,
  users AS u
WHERE u.Id = p.OwnerUserId
  AND u.Id = c.UserId
  AND u.Id = v.UserId
  AND c.Score = 0
  AND p.ViewCount >= 0
  AND u.Reputation <= 306
  AND u.UpVotes >= 0;
