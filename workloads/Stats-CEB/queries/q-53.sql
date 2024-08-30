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
  AND c.CreationDate <= CAST('2014-09-08 15:58:08' AS timestamp)
  AND p.ViewCount >= 0
  AND u.Reputation >= 1;
