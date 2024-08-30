SELECT COUNT(*)
FROM comments AS c,
  posts AS p,
  postLinks AS pl,
  votes AS v,
  badges AS b,
  users AS u
WHERE p.Id = c.PostId
  AND p.Id = pl.RelatedPostId
  AND p.Id = v.PostId
  AND u.Id = p.OwnerUserId
  AND u.Id = b.UserId
  AND u.Views <= 190
  AND u.CreationDate >= CAST('2010-07-20 08:05:39' AS timestamp)
  AND u.CreationDate <= CAST('2014-08-27 09:31:28' AS timestamp);
