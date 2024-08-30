SELECT COUNT(*)
FROM posts AS p, postLinks AS pl, users AS u
WHERE p.Id = pl.PostId
  AND p.OwnerUserId = u.Id
  AND p.CommentCount <= 17
  AND u.CreationDate <= CAST('2014-09-12 07:12:16' AS timestamp);
