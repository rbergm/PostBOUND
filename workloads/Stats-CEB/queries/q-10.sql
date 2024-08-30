SELECT COUNT(*)
FROM comments AS c, posts AS p, users AS u
WHERE c.UserId = u.Id
  AND u.Id = p.OwnerUserId
  AND c.CreationDate >= CAST('2010-08-05 00:36:02' AS timestamp)
  AND c.CreationDate <= CAST('2014-09-08 16:50:49' AS timestamp)
  AND p.ViewCount >= 0
  AND p.ViewCount <= 2897
  AND p.CommentCount >= 0
  AND p.CommentCount <= 16
  AND p.FavoriteCount >= 0
  AND p.FavoriteCount <= 10;
