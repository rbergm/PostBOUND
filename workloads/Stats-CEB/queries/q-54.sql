SELECT COUNT(*)
FROM comments AS c,
  postLinks AS pl,
  posts AS p,
  users AS u,
  badges AS b
WHERE p.Id = pl.RelatedPostId
  AND p.Id = c.PostId
  AND u.Id = b.UserId
  AND u.Id = p.OwnerUserId
  AND pl.LinkTypeId = 1
  AND pl.CreationDate >= CAST('2011-04-12 15:23:59' AS timestamp)
  AND p.Score = 1
  AND p.ViewCount >= 0
  AND p.FavoriteCount >= 0
  AND u.CreationDate >= CAST('2011-02-08 18:11:37' AS timestamp);
