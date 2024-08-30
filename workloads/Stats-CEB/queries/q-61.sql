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
  AND c.CreationDate >= CAST('2010-08-06 12:21:39' AS timestamp)
  AND c.CreationDate <= CAST('2014-09-11 20:55:34' AS timestamp)
  AND p.Score >= 0
  AND p.Score <= 13
  AND p.FavoriteCount >= 0
  AND pl.LinkTypeId = 1
  AND pl.CreationDate >= CAST('2011-03-11 18:50:29' AS timestamp)
  AND v.VoteTypeId = 2
  AND v.CreationDate <= CAST('2014-09-11 00:00:00' AS timestamp)
  AND u.Reputation >= 1
  AND u.CreationDate >= CAST('2011-02-17 03:42:02' AS timestamp)
  AND u.CreationDate <= CAST('2014-09-01 10:54:39' AS timestamp);
