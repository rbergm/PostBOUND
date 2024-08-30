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
  AND p.Score >= -2
  AND p.Score <= 23
  AND p.ViewCount <= 2432
  AND p.CommentCount = 0
  AND p.FavoriteCount >= 0
  AND u.Reputation >= 1
  AND u.Reputation <= 113
  AND u.Views >= 0
  AND u.Views <= 51;
