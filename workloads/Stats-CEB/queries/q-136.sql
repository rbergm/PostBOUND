SELECT COUNT(*)
FROM comments AS c,
  posts AS p,
  postHistory AS ph,
  badges AS b,
  users AS u
WHERE u.Id = ph.UserId
  AND u.Id = b.UserId
  AND u.Id = p.OwnerUserId
  AND u.Id = c.UserId
  AND c.CreationDate >= CAST('2010-08-19 09:33:49' AS timestamp)
  AND c.CreationDate <= CAST('2014-08-28 06:54:21' AS timestamp)
  AND p.PostTypeId = 1
  AND p.ViewCount >= 0
  AND p.ViewCount <= 25597
  AND p.CommentCount >= 0
  AND p.CommentCount <= 11
  AND p.FavoriteCount >= 0
  AND u.DownVotes <= 0
  AND u.UpVotes >= 0
  AND u.UpVotes <= 123;
